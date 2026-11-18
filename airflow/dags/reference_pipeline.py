"""Reference pipeline in Airflow: prepare -> train -> evaluate.

Started manually (UI or `airflow dags trigger`); there is no schedule. Each run
writes into its own new directory data/airflow_runs/<run ID>/ and calls the same
stage commands as dvc.yaml. A failed quality gate makes evaluate exit with
code 1, which fails the task and the DAG run.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task

from reference_stages import (
    PROJECT_DIRECTORY,
    QUALITY_GATES,
    RUNS_DIRECTORY,
    RUNTIME_PYTHON,
    STAGE_ORDER,
    run_directory_name,
    stage_arguments,
)

RUN_DIRECTORY = "{{ ti.xcom_pull(task_ids='create_run_directory')['run_directory'] }}"
CRITERIA = "{{ ti.xcom_pull(task_ids='create_run_directory')['criteria'] }}"
STAGE_ENVIRONMENT = {"PANDAS_COPY_ON_WRITE": "0", "PYTHONDONTWRITEBYTECODE": "1"}


@dag(
    dag_id="reference_pipeline",
    description="Reference stage pipeline with the team quality gate",
    schedule=None,
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"retries": 0},
    params={
        "quality_gate": Param(
            "team",
            type="string",
            enum=sorted(QUALITY_GATES),
            description="team: src/acceptance_criteria.json. failure_demo: minima no model can reach.",
        ),
    },
    tags=["phase-3", "pipeline"],
)
def reference_pipeline():
    @task
    def create_run_directory(params=None, run_id=None, dag=None):
        """Create the empty run directory and record where the run came from."""
        quality_gate = params["quality_gate"]
        relative_directory = f"{RUNS_DIRECTORY}/{run_directory_name(run_id)}"
        directory = Path(PROJECT_DIRECTORY) / relative_directory
        directory.mkdir()  # fails if the directory exists, so no run reuses outputs
        record = {
            "dag_id": dag.dag_id,
            "airflow_run_id": run_id,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
            "quality_gate": quality_gate,
            "criteria": QUALITY_GATES[quality_gate],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (directory / "airflow_run.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(f"Run directory {relative_directory}, source commit {record['source_commit']}.")
        return {"run_directory": relative_directory, "criteria": record["criteria"]}

    previous = create_run_directory()
    for stage in STAGE_ORDER:
        command = " ".join([RUNTIME_PYTHON, *stage_arguments(stage, RUN_DIRECTORY, CRITERIA)])
        current = BashOperator(
            task_id=stage,
            bash_command=command,
            cwd=PROJECT_DIRECTORY,
            env=STAGE_ENVIRONMENT,
            append_env=True,
        )
        previous >> current
        previous = current


reference_pipeline()
