"""Exercise descriptive evaluation using isolated fitted fixture artifacts."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import evaluate_run as evaluation


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / "fixture_run"
    (directory / "preprocessed").mkdir(parents=True)
    (directory / "model").mkdir()
    # This small, deterministic estimator is only a test fixture. The evaluator
    # does not fit or modify the supplied reference estimator or any run artifact.
    train = pd.DataFrame(np.zeros((6, 28)), columns=evaluation.FEATURES)
    train["place"] = [0, 0, 0, 0, 1, 1]
    targets = pd.DataFrame({"grav": [0, 0, 0, 0, 1, 1]})
    test = train.iloc[[0, 4, 5]].reset_index(drop=True)
    test_targets = pd.DataFrame({"grav": [0, 1, 1]})
    model = RandomForestClassifier(n_estimators=1, bootstrap=False, random_state=0, n_jobs=1)
    model.fit(train, targets.to_numpy().ravel())
    joblib.dump(model, directory / "model/trained_model.joblib")
    for name, frame in (("X_test", test), ("y_train", targets), ("y_test", test_targets)):
        frame.to_csv(directory / "preprocessed" / f"{name}.csv", index=False)
    validation = {
        "scope": "technical_data_and_model_validation", "validation_status": "passed",
        "contract_version": evaluation.CONTRACT_VERSION, "run_id": directory.name,
        "model_sha256": hashlib.sha256((directory / "model/trained_model.joblib").read_bytes()).hexdigest(),
        "approved_for_delivery": False,
    }
    (directory / "validation_report.json").write_text(json.dumps(validation), encoding="utf-8")
    return directory


def contents(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_cli_evaluates_real_fitted_fixture_and_preserves_inputs(run, tmp_path):
    before = contents(run)
    output = tmp_path / "evaluation.json"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", str(Path(evaluation.__file__)), "--run-dir", str(run), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"] == {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0}
    baseline = report["training_majority_baseline"]
    assert baseline["class_code"] == 0  # The held-out majority is 1.
    assert baseline["training_class_counts"] == {"0": 4, "1": 2}
    assert baseline["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert baseline["metrics"]["balanced_accuracy"] == 0.5
    assert baseline["metrics"]["macro_f1"] == 0.25
    assert report["test_rows"] == 3 and report["feature_count"] == 28
    assert report["quality_thresholds_enforced"] is False
    assert report["approved_for_delivery"] is False
    assert report["model_quality_approval"] == "NOT_ESTABLISHED"
    for recorded in report["inputs"].values():
        data = Path(recorded["path"]).read_bytes()
        assert recorded["sha256"] == hashlib.sha256(data).hexdigest()
        assert recorded["size_bytes"] == len(data)
    assert contents(run) == before


def test_model_mismatch_stops_before_deserialization(run, tmp_path):
    with (run / "model/trained_model.joblib").open("ab") as stream:
        stream.write(b"changed")
    output = tmp_path / "evaluation.json"
    with patch.object(evaluation.joblib, "load") as load:
        with pytest.raises(ValueError, match="Model bytes do not match"):
            evaluation.evaluate_run(run, output)
    load.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("validation_status", "failed"), ("run_id", "other_run"), ("approved_for_delivery", True)])
def test_unmatched_technical_validation_is_rejected(run, tmp_path, field, value):
    path = run / "validation_report.json"
    validation = json.loads(path.read_text())
    validation[field] = value
    path.write_text(json.dumps(validation), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="matching passed technical validation"):
        evaluation.evaluate_run(run, output)
    assert not output.exists()


@pytest.mark.parametrize("case", ["feature_order", "row_count", "nonfinite", "target_code", "single_class"])
def test_invalid_held_out_inputs_produce_no_report(run, tmp_path, case):
    feature_path, target_path = run / "preprocessed/X_test.csv", run / "preprocessed/y_test.csv"
    features, targets = pd.read_csv(feature_path), pd.read_csv(target_path)
    if case == "feature_order":
        features = features[list(reversed(evaluation.FEATURES))]
    elif case == "row_count":
        targets = targets.iloc[:1]
    elif case == "nonfinite":
        features.iloc[0, 0] = np.inf
    elif case == "target_code":
        targets.iloc[0, 0] = 2
    else:
        targets.iloc[:, 0] = 0
    features.to_csv(feature_path, index=False)
    targets.to_csv(target_path, index=False)
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError):
        evaluation.evaluate_run(run, output)
    assert not output.exists()


def test_existing_output_is_not_overwritten(run, tmp_path):
    output = tmp_path / "evaluation.json"
    output.write_bytes(b"PRESERVED")
    with patch.object(evaluation.joblib, "load") as load:
        with pytest.raises(FileExistsError):
            evaluation.evaluate_run(run, output)
    load.assert_not_called()
    assert output.read_bytes() == b"PRESERVED"


def test_input_change_during_prediction_is_rejected(run, tmp_path):
    original = RandomForestClassifier.predict

    def changed_input(model, features):
        predictions = original(model, features)
        with (run / "preprocessed/y_train.csv").open("ab") as stream:
            stream.write(b"0\n")
        return predictions

    output = tmp_path / "evaluation.json"
    with patch.object(RandomForestClassifier, "predict", changed_input):
        with pytest.raises(ValueError, match="input changed during execution"):
            evaluation.evaluate_run(run, output)
    assert not output.exists()


def test_training_majority_tie_uses_lowest_code(run, tmp_path):
    pd.DataFrame({"grav": [0, 1]}).to_csv(run / "preprocessed/y_train.csv", index=False)
    output = evaluation.evaluate_run(run, tmp_path / "evaluation.json")
    assert json.loads(output.read_text())["training_majority_baseline"]["class_code"] == 0


def test_low_metrics_do_not_become_quality_or_delivery_approval(run, tmp_path):
    with patch.object(RandomForestClassifier, "predict", return_value=np.zeros(3, dtype=int)):
        output = evaluation.evaluate_run(run, tmp_path / "evaluation.json")
    report = json.loads(output.read_text())
    assert report["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert report["evaluation_status"] == "completed"
    assert report["model_quality_approval"] == "NOT_ESTABLISHED"
    assert report["approved_for_delivery"] is False


def test_matching_digest_does_not_bypass_model_feature_contract(run, tmp_path):
    model_path = run / "model/trained_model.joblib"
    model = joblib.load(model_path)
    model.feature_names_in_ = np.array(list(reversed(evaluation.FEATURES)))
    joblib.dump(model, model_path)
    validation_path = run / "validation_report.json"
    validation = json.loads(validation_path.read_text())
    validation["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="reference contract"):
        evaluation.evaluate_run(run, output)
    assert not output.exists()
