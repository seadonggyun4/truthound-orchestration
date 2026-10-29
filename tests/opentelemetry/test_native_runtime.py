"""Native SDK regressions: bounded parsing and real trace/metric serialization."""

import subprocess
import sys

import pytest


def test_native_context_bridge_bounds_malformed_tracestate(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import logging
import time
from common.opentelemetry.bridge.context import ContextBridge
from opentelemetry.trace import get_current_span
logging.disable(logging.CRITICAL)
bridge = ContextBridge()
parent = '00-12345678901234567890123456789012-1234567890123456-01'
valid = bridge.extract({'traceparent': parent, 'tracestate': 'vendor=value'})
assert get_current_span(valid).get_span_context().trace_state.get('vendor') == 'value'
start = time.process_time()
context = bridge.extract({'traceparent': parent, 'tracestate': 'a=b' + ' ' * 200000 + 'x'})
assert time.process_time() - start < 1.0, 'tracestate parsing consumed excessive CPU'
span = get_current_span(context).get_span_context()
assert span.is_valid
assert len(span.trace_state) == 0
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("protocol", ["grpc", "http/protobuf"])
@pytest.mark.parametrize("factory_surface", ["provider", "public"])
def test_native_sdk_exports_traces_and_metrics(protocol, factory_surface):
    import gzip
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    import grpc
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2 as metrics_pb
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as traces_pb
    from opentelemetry.sdk.metrics.export import MetricExportResult
    from opentelemetry.sdk.trace.export import SpanExportResult

    from common.opentelemetry.config import OTelConfig, OTLPExporterConfig
    from common.opentelemetry.providers.meter import OTelMeterProvider, _create_otlp_metric_exporter
    from common.opentelemetry.providers.tracer import OTelTracerProvider, _create_otlp_span_exporter
    from common.opentelemetry.types import ExporterType, OTLPCompression, OTLPProtocol

    if factory_surface == "public":
        from common.opentelemetry.exporters import create_metric_exporter, create_span_exporter

        span_factory = partial(create_span_exporter, ExporterType.OTLP)
        metric_factory = partial(create_metric_exporter, ExporterType.OTLP)
    else:
        span_factory = _create_otlp_span_exporter
        metric_factory = _create_otlp_metric_exporter

    captured = {}
    if protocol == "grpc":
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        for kind, service, request_type, response_type in [
            (
                "traces",
                "trace.v1.TraceService",
                traces_pb.ExportTraceServiceRequest,
                traces_pb.ExportTraceServiceResponse,
            ),
            (
                "metrics",
                "metrics.v1.MetricsService",
                metrics_pb.ExportMetricsServiceRequest,
                metrics_pb.ExportMetricsServiceResponse,
            ),
        ]:

            def receive(request, context, kind=kind, response_type=response_type):
                captured[kind] = request
                return response_type()

            server.add_generic_rpc_handlers(
                (
                    grpc.method_handlers_generic_handler(
                        "opentelemetry.proto.collector." + service,
                        {
                            "Export": grpc.unary_unary_rpc_method_handler(
                                receive,
                                request_deserializer=request_type.FromString,
                                response_serializer=response_type.SerializeToString,
                            )
                        },
                    ),
                )
            )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        endpoint = f"http://127.0.0.1:{port}"
    else:

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                kind = self.path.rsplit("/", 1)[-1]
                message_type = (
                    traces_pb.ExportTraceServiceRequest
                    if kind == "traces"
                    else metrics_pb.ExportMetricsServiceRequest
                )
                captured[kind] = message_type.FromString(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Receiver)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}"

    config = OTelConfig(traces_exporter=ExporterType.MEMORY, metrics_exporter=ExporterType.MEMORY)
    with OTelTracerProvider(config) as tracer, OTelMeterProvider(config) as meter:
        try:
            with tracer.get_tracer("security-regression").start_as_current_span("quality-check"):
                meter.get_meter("security-regression").create_counter("checks").add(1)
            spans = tracer.get_collected_spans()
            # The public wrapper exposes ResourceMetrics; the reader holds the SDK envelope.
            assert meter.get_collected_metrics()
            data = meter._reader.get_metrics_data()
            for kind, factory, payload, expected in [
                ("traces", span_factory, spans, SpanExportResult.SUCCESS),
                ("metrics", metric_factory, data, MetricExportResult.SUCCESS),
            ]:
                exporter = factory(
                    OTLPExporterConfig(
                        endpoint=endpoint if protocol == "grpc" else endpoint + "/v1/" + kind,
                        protocol=OTLPProtocol(protocol),
                        insecure=True,
                        compression=OTLPCompression.GZIP,
                    )
                )
                try:
                    assert exporter.export(payload) == expected
                finally:
                    exporter.shutdown()
            assert (
                captured["traces"].resource_spans[0].scope_spans[0].spans[0].name == "quality-check"
            )
            assert (
                captured["metrics"]
                .resource_metrics[0]
                .scope_metrics[0]
                .metrics[0]
                .sum.data_points[0]
                .as_int
                == 1
            )
        finally:
            if protocol == "grpc":
                server.stop(0).wait()
            else:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
