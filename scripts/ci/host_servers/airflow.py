"""Isolated real-host execution regression; invoked by verify_host_server.py."""

import importlib.metadata
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
dags = work / "dags"
dags.mkdir(exist_ok=True)
(work / "good.csv").write_text("id\n1\n2\n3\n")
(work / "bad.csv").write_text("id,value\n1,10\n,20\n3,30\n")
(dags / "quality.py").write_text(f"""from airflow import DAG
from datetime import datetime
from truthound_airflow import DataQualityCheckOperator
with DAG('truthound_core33_server', start_date=datetime(2024,1,1),schedule=None,catchup=False) as dag:
    good=DataQualityCheckOperator(task_id='good',data_path={str(work / "good.csv")!r})
    bad=DataQualityCheckOperator(task_id='bad',data_path={str(work / "bad.csv")!r})
""")
env = dict(
    os.environ,
    PATH=str(pathlib.Path(py).parent) + os.pathsep + os.environ["PATH"],
    AIRFLOW_HOME=str(work),
    AIRFLOW__CORE__LOAD_EXAMPLES="false",
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION="false",
    AIRFLOW__CORE__EXECUTOR="SequentialExecutor",
    AIRFLOW__CORE__PARALLELISM="2",
    AIRFLOW__CORE__MP_START_METHOD="spawn",
    AIRFLOW__CORE__EXECUTE_TASKS_NEW_PYTHON_INTERPRETER="true",
    AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL="2",
    AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL="2",
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
)
is_v3 = int(importlib.metadata.version("apache-airflow").split(".")[0]) >= 3
if is_v3:
    env.update(
        AIRFLOW__CORE__EXECUTOR="LocalExecutor",
        AIRFLOW__CORE__EXECUTION_API_SERVER_URL=f"http://127.0.0.1:{port}/execution/",
        AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS="true",
    )
if sys.platform == "darwin" and not is_v3:
    # Airflow 2.6's StandardTaskRunner always forks on POSIX. Its native
    # setproctitle call can deadlock after fork on modern macOS. Use the host's
    # supported custom task-runner extension to launch a fresh interpreter.
    (work / "verification_task_runner.py").write_text(
        "from airflow.task.task_runner.standard_task_runner import StandardTaskRunner\n"
        "class ExecTaskRunner(StandardTaskRunner):\n"
        "    def start(self):\n"
        "        self.process = self._start_by_exec()\n"
    )
    env["AIRFLOW__CORE__TASK_RUNNER"] = "verification_task_runner.ExecTaskRunner"
    env["PYTHONPATH"] = str(work)
procs: list[subprocess.Popen[bytes]] = []


def run(args: list[str], label: str) -> None:
    with (work / (label + ".log")).open("w") as f:
        subprocess.run(
            [py, "-m", "airflow", *args],
            env=env,
            cwd=work,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )


def start(args: list[str], label: str) -> None:
    f = (work / (label + ".log")).open("w")
    p = subprocess.Popen(
        [py, "-m", "airflow", *args],
        env=env,
        cwd=work,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs.append(p)


try:
    run(["db", "migrate" if is_v3 else "init"], "db-init")
    if is_v3:
        start(["scheduler"], "scheduler")
        start(["dag-processor"], "dag-processor")
        start(["api-server", "--host", "127.0.0.1", "--port", port, "--workers", "1"], "webserver")
    else:
        start(["scheduler", "--skip-serve-logs"], "scheduler")
        start(
            ["webserver", "--hostname", "127.0.0.1", "--port", port, "--workers", "1"], "webserver"
        )
    health = "/api/v2/monitor/health" if is_v3 else "/health"
    for _ in range(600):
        if any(p.poll() is not None for p in procs):
            raise RuntimeError("A host service exited; inspect service logs")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}" + health, timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(1)
    else:
        raise TimeoutError("Airflow HTTP health")
    code = """
import time,json
from airflow.models import DagModel,DagRun,TaskInstance
from airflow.utils.session import create_session
for _ in range(600):
 with create_session() as s:
  if s.query(DagModel).filter(DagModel.dag_id=='truthound_core33_server').first():break
 time.sleep(1)
else:raise TimeoutError('DAG discovery')
"""
    subprocess.run([py, "-c", code], env=env, check=True, timeout=650)
    run(
        ["dags", "trigger", "truthound_core33_server", "--run-id", "core33-verification"], "trigger"
    )
    code += """
for _ in range(600):
 with create_session() as s:
  tis=s.query(TaskInstance).filter(TaskInstance.dag_id=='truthound_core33_server',TaskInstance.run_id=='core33-verification').all()
  states={t.task_id:t.state for t in tis}
  if states.get('good')=='success' and states.get('bad')=='failed':
   xcom={t.task_id:t.xcom_pull(task_ids=t.task_id,key='data_quality_result',session=s) for t in tis}
   assert xcom['good']['is_success'] is True,xcom
   assert xcom['bad']['is_success'] is False,xcom
   print(json.dumps({'states':states,'xcom_success':{k:v['is_success'] for k,v in xcom.items()}}));break
 time.sleep(1)
else:raise TimeoutError(states)
"""
    if is_v3:
        code = code.replace(
            "from airflow.models import DagModel,DagRun,TaskInstance",
            "from airflow.models import DagModel,DagRun,TaskInstance\nfrom airflow.models.xcom import XComModel",
        )
        code = code.replace(
            "xcom={t.task_id:t.xcom_pull(task_ids=t.task_id,key='data_quality_result',session=s) for t in tis}",
            "xcom={x.task_id:x.value for x in s.query(XComModel).filter(XComModel.dag_id=='truthound_core33_server',XComModel.run_id=='core33-verification',XComModel.key=='data_quality_result').all()}",
        )
    with (work / "results.log").open("w") as f:
        subprocess.run(
            [py, "-c", code], env=env, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1250
        )
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
