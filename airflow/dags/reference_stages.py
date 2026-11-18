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
    # Minima no model can reach; shows how a failed gate looks in Airflow.
    "failure_demo": "airflow/gate_failure_demo_criteria.json",
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
