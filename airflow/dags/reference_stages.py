"""Stage commands for the Airflow pipeline, kept free of Airflow imports.

The commands are the ones in dvc.yaml and compose.pipeline.yaml; only the
interpreter and the run directory differ. src/test_airflow_dag.py checks this.
"""

import re

STAGE_ORDER = ("prepare", "train", "evaluate")

# Relative to PROJECT_DIRECTORY inside the Airflow image (airflow/Dockerfile).
PROJECT_DIRECTORY = "/opt/airflow/project"
RUNTIME_PYTHON = "/opt/runtime/bin/python"
RUNS_DIRECTORY = "data/airflow_runs"

QUALITY_GATES = {
    "team": "src/acceptance_criteria.json",
    # Minima of 1.0; with the reference data the F1 gain is at most ~0.604, so the gate fails.
    "failure_demo": "airflow/gate_failure_demo_criteria.json",
}

# MLflow target of the task `track` (src/tracking/track_stage_run.py). `local`
# needs the MLflow service of compose.tracking.yaml and compose.airflow.tracking.yaml;
# `dagshub` needs MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD; `none` skips it.
TRACKING_TARGETS = {
    "local": "http://mlflow:5000",
    "dagshub": "https://dagshub.com/SiR1979/prediction_severity_car_accidents.mlflow",
    "none": "",
}

RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")


def stage_arguments(stage, run_directory, criteria_path):
    """Arguments after the interpreter, as in the `cmd` entries of dvc.yaml."""
    arguments = {
        "prepare": ["prepare", "--raw-dir", "data/raw", "--run-dir", run_directory],
        "train": ["train", "--run-dir", run_directory],
        "evaluate": ["evaluate", "--run-dir", run_directory, "--criteria", criteria_path],
    }[stage]
    return ["-B", "src/pipeline_stages.py", *arguments]


def run_directory_name(run_id):
    """Map an Airflow run ID to one safe directory name below RUNS_DIRECTORY."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Run ID {run_id!r} gives no safe run directory name.")
    return name


def track_arguments(run_directory, criteria_path, tracking_uri):
    """Arguments after the interpreter for the MLflow recorder; IDs come from the environment."""
    return [
        "-B", "-m", "src.tracking.track_stage_run",
        "--run-dir", run_directory, "--raw-dir", "data/raw", "--criteria", criteria_path,
        "--orchestrator", "airflow", "--orchestrator-run-id", '"$AIRFLOW_RUN_ID"',
        "--source-commit", '"$SOURCE_COMMIT"', "--tracking-uri", tracking_uri,
    ]
