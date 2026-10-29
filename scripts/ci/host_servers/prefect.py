"""Isolated real-host execution regression; invoked by verify_host_server.py."""

import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request


root = pathlib.Path(sys.argv[1])
name = "runtime"
py = sys.executable
port = sys.argv[2]
work = root / name
work.mkdir(exist_ok=True)
env = dict(
    os.environ,
    PATH=str(pathlib.Path(py).parent) + os.pathsep + os.environ["PATH"],
    PREFECT_HOME=str(work),
    PREFECT_API_URL=f"http://127.0.0.1:{port}/api",
    PREFECT_SERVER_ANALYTICS_ENABLED="false",
    PREFECT_LOGGING_LEVEL="INFO",
)
procs = []


def start(args, label):
    f = (work / (label + ".log")).open("w")
    p = subprocess.Popen(
        (
            [str(pathlib.Path(py).with_name("prefect")), *args[2:]]
            if args[:2] == ["-m", "prefect"]
            else [py, *args]
        ),
        env=env,
        cwd=root,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs.append(p)
    return p


try:
    p = start(["-m", "prefect", "server", "start", "--host", "127.0.0.1", "--port", port], "server")
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError("server exited")
        try:
            with urllib.request.urlopen(env["PREFECT_API_URL"] + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("server health")
    p = start(["prefect_flow.py"], "worker")
    code = """
import asyncio,json
from prefect.client.orchestration import get_client
async def main():
 async with get_client() as c:
  for _ in range(600):
   try:
    d=await c.read_deployment_by_name('truthound-core33-server/verification');break
   except Exception:await asyncio.sleep(1)
  else:raise RuntimeError('deployment missing')
  results=[]
  for bad in (False,True):
   r=await c.create_flow_run_from_deployment(d.id,parameters={'bad':bad})
   for _ in range(600):
    r=await c.read_flow_run(r.id)
    if r.state.is_final():break
    await asyncio.sleep(1)
   assert (r.state.is_failed() if bad else r.state.is_completed()),str(r.state)
   results.append({'bad':bad,'run_id':str(r.id),'state':str(r.state.type)})
  print(json.dumps(results))
asyncio.run(main())
"""
    r = subprocess.run(
        [py, "-c", code],
        env=env,
        cwd=root,
        stdout=(work / "runs.log").open("w"),
        stderr=subprocess.STDOUT,
        timeout=1900,
    )
    if r.returncode:
        raise RuntimeError("execution failed")
    print(name, "PASSED")
finally:
    for p in reversed(procs):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
