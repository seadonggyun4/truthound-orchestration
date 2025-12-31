"""Tests for OpenTelemetry types."""

import pytest

from common.opentelemetry.types import (
    ExporterType,
    OTLPCompression,
    OTLPProtocol,
    SamplingDecision,
)


class TestOTLPProtocol:
    """Tests for OTLPProtocol enum."""

    def test_grpc(self):
        """Test gRPC protocol."""
        assert OTLPProtocol.GRPC.value == "grpc"

    def test_http_protobuf(self):
        """Test HTTP protobuf protocol."""
        assert OTLPProtocol.HTTP_PROTOBUF.value == "http/protobuf"

    def test_http_json(self):
        """Test HTTP JSON protocol."""
        assert OTLPProtocol.HTTP_JSON.value == "http/json"


class TestOTLPCompression:
    """Tests for OTLPCompression enum."""

    def test_none(self):
        """Test no compression."""
        assert OTLPCompression.NONE.value == "none"

    def test_gzip(self):
        """Test gzip compression."""
        assert OTLPCompression.GZIP.value == "gzip"


class TestExporterType:
    """Tests for ExporterType enum."""

    def test_otlp(self):
        """Test OTLP exporter."""
        assert ExporterType.OTLP.value == "otlp"

    def test_console(self):
        """Test console exporter."""
        assert ExporterType.CONSOLE.value == "console"

    def test_memory(self):
        """Test memory exporter."""
        assert ExporterType.MEMORY.value == "memory"

    def test_none(self):
        """Test no-op exporter."""
        assert ExporterType.NONE.value == "none"


class TestSamplingDecision:
    """Tests for SamplingDecision enum."""

    def test_drop(self):
        """Test drop decision."""
        assert SamplingDecision.DROP.value == "drop"

    def test_record_only(self):
        """Test record only decision."""
        assert SamplingDecision.RECORD_ONLY.value == "record_only"

    def test_record_and_sample(self):
        """Test record and sample decision."""
        assert SamplingDecision.RECORD_AND_SAMPLE.value == "record_and_sample"
