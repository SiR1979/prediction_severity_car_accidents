"""Reference pipeline in Airflow: prepare -> train -> evaluate -> track -> finish.

Started manually (UI or `airflow dags trigger`); there is no schedule. Each run
writes into its own new directory data/airflow_runs/<run ID>/ and calls the same
stage commands as dvc.yaml. A failed quality gate makes evaluate exit with
code 1, which fails the task and the DAG run.

`track` runs after evaluate even when a stage failed and records the run in
MLflow (src/tracking/track_stage_run.py), tagged with the Airflow run ID.
`finish` needs evaluate and track to succeed, so the DAG run fails when the
gate fails or the run could not be recorded.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Param, dag, task

from reference_stages import (
    PROJECT_DIRECTORY,
    QUALITY_GATES,
    RUNS_DIRECTORY,
    RUNTIME_PYTHON,
    STAGE_ORDER,
    TRACKING_TARGETS,
    run_directory_name,
    stage_arguments,
    track_arguments,
)

RUN_DIRECTORY = "{{ ti.xcom_pull(task_ids='create_run_directory')['run_directory'] }}"
CRITERIA = "{{ ti.xcom_pull(task_ids='create_run_directory')['criteria'] }}"
TRACKING_URI = "{{ ti.xcom_pull(task_ids='create_run_directory')['tracking_uri'] }}"
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
            description=(
                "team: src/acceptance_criteria.json. failure_demo: all minima 1.0, only to show "
                "a failed gate. With the reference data the training majority baseline has a "
                "macro F1 of about 0.396, so the macro F1 gain can be at most about 0.604; "
                "the gain check (minimum 1.0) fails for every model, and so does the gate."
            ),
        ),
        "tracking_target": Param(
            "local",
            type="string",
            enum=sorted(TRACKING_TARGETS),
            description="MLflow target of the task track: local (compose.tracking.yaml), dagshub, or none.",
        ),
    },
    tags=["phase-3", "pipeline"],
)
def reference_pipeline():
    @task
    def create_run_directory(params=None, run_id=None, dag=None):
        """Create the empty run directory and record where the run came from."""
        quality_gate = params["quality_gate"]
        tracking_target = params["tracking_target"]
        relative_directory = f"{RUNS_DIRECTORY}/{run_directory_name(run_id)}"
        directory = Path(PROJECT_DIRECTORY) / relative_directory
        directory.mkdir()  # fails if the directory exists, so no run reuses outputs
        record = {
            "dag_id": dag.dag_id,
            "airflow_run_id": run_id,
            "source_commit": os.environ.get("SOURCE_COMMIT", "unknown"),
            "quality_gate": quality_gate,
            "criteria": QUALITY_GATES[quality_gate],
            "tracking_target": tracking_target,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (directory / "airflow_run.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(f"Run directory {relative_directory}, source commit {record['source_commit']}.")
        return {"run_directory": relative_directory, "criteria": record["criteria"],
                "tracking_uri": TRACKING_TARGETS[tracking_target]}

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

    track_command = " ".join([RUNTIME_PYTHON, *track_arguments(RUN_DIRECTORY, CRITERIA, TRACKING_URI)])
    track = BashOperator(
        task_id="track",
        bash_command=(
            "{% if ti.xcom_pull(task_ids='create_run_directory')['tracking_uri'] %}" + track_command
            + "{% else %}echo 'MLflow tracking is off for this run (tracking_target=none).'{% endif %}"
        ),
        cwd=PROJECT_DIRECTORY,
        env={**STAGE_ENVIRONMENT, "AIRFLOW_RUN_ID": "{{ run_id }}"},
        append_env=True,
        trigger_rule="all_done",  # also record runs whose gate or stage failed
    )
    finish = EmptyOperator(task_id="finish")  # all_success: fails the run if evaluate or track failed
    previous >> track >> finish
    previous >> finish


reference_pipeline()
