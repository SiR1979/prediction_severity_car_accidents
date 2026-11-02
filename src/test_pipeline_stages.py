"""Verify the stage entry points and their orchestration files with doubles."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

import pipeline_stages as stages
from reference_contract import FEATURES


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_BYTES = b"Opaque test artifact; never deserialize or serve these bytes."


def prepare_double(*arguments):
    directory = arguments[-1]
    directory.mkdir(parents=True)
    for partition in ("train", "test"):
        pd.DataFrame(np.ones((2, 28)), columns=FEATURES).to_csv(directory / f"X_{partition}.csv", index=False)
        pd.DataFrame({"grav": [0, 1]}).to_csv(directory / f"y_{partition}.csv", index=False)


def train_double(directory, destination):
    destination.parent.mkdir()
    destination.write_bytes(ARTIFACT_BYTES)
    return destination


def model_double():
    model = RandomForestClassifier()
    model.feature_names_in_ = np.array(FEATURES)
    model.n_features_in_ = 28
    model.classes_ = np.array([0, 1])
    model.fit = Mock(side_effect=AssertionError("Real training is forbidden in this suite."))
    return model


@pytest.fixture
def raw(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    for name in stages.RAW_FILENAMES:
        (directory / name).write_bytes(b"Opaque input fixture; preprocessing is mocked.\n")
    return directory


@pytest.fixture
def doubles():
    with patch.object(stages, "process_data", side_effect=prepare_double) as process, \
            patch.object(stages, "train_model", side_effect=train_double) as train, \
            patch("run_pipeline.joblib.load", return_value=model_double()):
        yield process, train


def test_prepare_and_train_write_the_reports_evaluation_expects(raw, tmp_path, doubles):
    run = tmp_path / "pipeline"
    stages.prepare(raw, run)
    assert {path.name for path in run.iterdir()} == {"preprocessed", "data_validation.json"}
    report_path = stages.train(run)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation_status"] == "passed"
    assert report["run_id"] == "pipeline"
    assert report["model_sha256"] == hashlib.sha256(ARTIFACT_BYTES).hexdigest()
    assert report["data_validation"] == json.loads((run / "data_validation.json").read_text(encoding="utf-8"))
    assert report["approved_for_delivery"] is False


def test_prepare_never_reuses_existing_outputs(raw, tmp_path, doubles):
    run = tmp_path / "pipeline"
    stages.prepare(raw, run)
    with pytest.raises(FileExistsError):
        stages.prepare(raw, run)


def test_run_directory_inside_raw_is_rejected_before_processing(raw, doubles):
    process, _ = doubles
    with pytest.raises(ValueError, match="outside the raw directory"):
        stages.prepare(raw, raw / "pipeline")
    process.assert_not_called()


def test_train_requires_the_prepare_outputs(tmp_path, doubles):
    _, train = doubles
    (tmp_path / "pipeline").mkdir()
    with pytest.raises(FileNotFoundError):
        stages.train(tmp_path / "pipeline")
    train.assert_not_called()


def test_evaluate_runs_the_gate_on_the_new_evaluation(tmp_path):
    run = tmp_path / "pipeline"
    run.mkdir()
    gate_report = {"gate_status": "passed"}
    with patch.object(stages, "evaluate_run", return_value=run / "evaluation_report.json") as evaluate, \
            patch.object(stages, "accept_run", return_value=gate_report) as accept:
        assert stages.evaluate(run, "criteria.json") == gate_report
    evaluate.assert_called_once_with(run.resolve(), run.resolve() / "evaluation_report.json")
    accept.assert_called_once_with(run / "evaluation_report.json", "criteria.json",
                                   run.resolve() / "acceptance_report.json", run.resolve() / "metrics.json")


@pytest.mark.parametrize("gate_status, code", [("passed", 0), ("failed", 1)])
def test_cli_exit_code_follows_the_gate(tmp_path, gate_status, code):
    argv = ["pipeline_stages.py", "evaluate", "--run-dir", str(tmp_path), "--criteria", "criteria.json"]
    with patch.object(stages, "evaluate", return_value={"gate_status": gate_status}), patch("sys.argv", argv):
        assert stages.main() == code


def test_stage_errors_return_a_distinct_exit_code(tmp_path):
    argv = ["pipeline_stages.py", "train", "--run-dir", str(tmp_path / "missing")]
    with patch("sys.argv", argv):
        assert stages.main() == 2


def test_dvc_stages_and_compose_services_run_the_same_commands():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    compose = yaml.safe_load((ROOT / "compose.pipeline.yaml").read_text(encoding="utf-8"))["services"]
    assert list(dvc) == list(compose) == ["prepare", "train", "evaluate"]
    for name, stage in dvc.items():
        assert stage["cmd"].split() == compose[name]["command"]
    assert compose["train"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
    assert compose["evaluate"]["depends_on"]["train"]["condition"] == "service_completed_successfully"
    for service in compose.values():
        assert service["network_mode"] == "none" and service["read_only"] is True
        raw_mounts = [mount for mount in service["volumes"] if mount["target"] == "/app/data/raw"]
        assert all(mount["read_only"] for mount in raw_mounts)
