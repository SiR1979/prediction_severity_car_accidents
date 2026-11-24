"""Check the Airflow DAG against dvc.yaml and the Airflow Compose files.

The command and Compose checks need no Airflow. The DAG checks import Airflow
and run in the Airflow image (CI job "Airflow DAG"); elsewhere they are skipped.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "airflow" / "dags"
sys.path.insert(0, str(DAGS))

import reference_stages as stages  # noqa: E402


def test_stage_commands_match_dvc():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    assert tuple(dvc) == stages.STAGE_ORDER
    for name, stage in dvc.items():
        interpreter, *arguments = stage["cmd"].split()
        assert interpreter == "python"
        assert stages.stage_arguments(name, "data/pipeline", "src/acceptance_criteria.json") == arguments


def test_team_gate_is_the_default_criteria_and_the_demo_can_only_fail():
    assert stages.QUALITY_GATES["team"] == "src/acceptance_criteria.json"
    team = json.loads((ROOT / stages.QUALITY_GATES["team"]).read_text(encoding="utf-8"))
    demo = json.loads((ROOT / stages.QUALITY_GATES["failure_demo"]).read_text(encoding="utf-8"))
    assert set(demo) == set(team)
    assert "demo" in demo["criteria_version"]
    # The macro F1 gain is at most 1 minus a positive baseline F1, so it never reaches 1.
    assert demo["min_macro_f1_gain_over_training_majority"] == 1.0
    for key in team:
        if key.startswith("min_"):
            assert demo[key] >= team[key]


@pytest.mark.parametrize("run_id, name", [
    ("manual__2026-11-18T10:15:00.123456+00:00", "manual__2026-11-18T10-15-00.123456-00-00"),
    ("demo_run", "demo_run"),
    ("../escape", "..-escape"),
])
def test_run_ids_map_to_one_directory_level(run_id, name):
    if name.startswith("."):
        with pytest.raises(ValueError):
            stages.run_directory_name(run_id)
    else:
        assert stages.run_directory_name(run_id) == name


def test_track_command_uses_the_shared_recorder_for_airflow():
    arguments = stages.track_arguments("RUN", "CRITERIA", "URI")
    assert arguments[:3] == ["-B", "-m", "src.tracking.track_stage_run"]
    options = dict(zip(arguments[3::2], arguments[4::2]))
    assert options == {"--run-dir": "RUN", "--raw-dir": "data/raw", "--criteria": "CRITERIA",
                       "--orchestrator": "airflow", "--orchestrator-run-id": '"$AIRFLOW_RUN_ID"',
                       "--source-commit": '"$SOURCE_COMMIT"', "--tracking-uri": "URI"}
    assert stages.TRACKING_TARGETS["local"] == "http://mlflow:5000"
    assert stages.TRACKING_TARGETS["dagshub"].endswith("/SiR1979/prediction_severity_car_accidents.mlflow")
    assert stages.TRACKING_TARGETS["none"] == ""


def test_tracking_overlay_joins_the_mlflow_network_and_passes_credentials_by_name():
    overlay = yaml.safe_load((ROOT / "compose.airflow.tracking.yaml").read_text(encoding="utf-8"))
    scheduler = overlay["services"]["airflow-scheduler"]
    assert set(overlay["services"]) == {"airflow-scheduler"}
    assert scheduler["networks"] == ["default", "tracking"]
    assert overlay["networks"]["tracking"] == {"external": True, "name": "car-accident_default"}
    for name, value in scheduler["environment"].items():
        assert value == f"${{{name}:-}}"


def test_compose_mounts_raw_read_only_and_exposes_the_ui_locally():
    compose = yaml.safe_load((ROOT / "compose.airflow.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    scheduler = services["airflow-scheduler"]
    assert scheduler["environment"]["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
    mounts = {mount["target"]: mount for mount in scheduler["volumes"] if isinstance(mount, dict)}
    project = stages.PROJECT_DIRECTORY
    assert mounts[f"{project}/data/raw"]["read_only"] is True
    assert mounts[f"{project}/{stages.RUNS_DIRECTORY}"]["source"] == f"./{stages.RUNS_DIRECTORY}"
    assert services["airflow-apiserver"]["ports"] == ["127.0.0.1:8081:8080"]


@pytest.fixture(scope="module")
def dag():
    if os.environ.get("REQUIRE_AIRFLOW") != "1":  # set in the CI job "Airflow DAG"
        pytest.importorskip("airflow.sdk", reason="Airflow is only installed in the Airflow image.")
    from airflow.dag_processing.dagbag import DagBag

    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    return bag.dags["reference_pipeline"]


def test_dag_runs_the_stages_in_order_after_creating_the_run_directory(dag):
    assert dag.schedule is None or str(dag.timetable.summary) == "None"
    assert dag.max_active_runs == 1
    chain = ["create_run_directory", *stages.STAGE_ORDER, "track"]
    assert set(dag.task_ids) == {*chain, "finish"}
    for upstream, downstream in zip(chain, chain[1:]):
        assert dag.get_task(downstream).upstream_task_ids == {upstream}


def test_track_also_runs_after_a_failed_stage_and_finish_carries_the_result(dag):
    assert dag.get_task("track").trigger_rule == "all_done"
    finish = dag.get_task("finish")
    assert finish.trigger_rule == "all_success"
    assert finish.upstream_task_ids == {"evaluate", "track"}
    assert [task.task_id for task in dag.leaves] == ["finish"]


def test_track_records_the_run_with_its_airflow_run_id(dag):
    import reference_pipeline as module

    task = dag.get_task("track")
    assert task.cwd == stages.PROJECT_DIRECTORY
    assert task.env["AIRFLOW_RUN_ID"] == "{{ run_id }}"
    command = " ".join([stages.RUNTIME_PYTHON, *stages.track_arguments(module.RUN_DIRECTORY, module.CRITERIA, module.TRACKING_URI)])
    assert command in task.bash_command
    assert "tracking_target=none" in task.bash_command
    assert dag.params["tracking_target"] == "local"


def test_dag_tasks_call_the_stage_commands_with_the_runtime_python(dag):
    import reference_pipeline as module

    for template, key in ((module.RUN_DIRECTORY, "run_directory"), (module.CRITERIA, "criteria")):
        assert template == f"{{{{ ti.xcom_pull(task_ids='create_run_directory')['{key}'] }}}}"
    for stage in stages.STAGE_ORDER:
        task = dag.get_task(stage)
        assert task.retries == 0
        assert task.cwd == stages.PROJECT_DIRECTORY
        assert task.env["PANDAS_COPY_ON_WRITE"] == "0"
        command = task.bash_command.replace(module.RUN_DIRECTORY, "RUN").replace(module.CRITERIA, "CRITERIA")
        interpreter, *arguments = command.split()
        assert interpreter == stages.RUNTIME_PYTHON
        assert arguments == stages.stage_arguments(stage, "RUN", "CRITERIA")


def test_dag_defaults_to_the_team_gate(dag):
    assert dag.params["quality_gate"] == "team"
