"""Check the team quality gate with isolated evaluation report fixtures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import accept_run as gate


CRITERIA_FILE = Path(gate.__file__).with_name("acceptance_criteria.json")


def evaluation_report(macro_f1=0.74, balanced_accuracy=0.73, baseline_macro_f1=0.40):
    return {
        "schema_name": "reference_run_evaluation_v1",
        "evaluation_status": "completed",
        "run_id": "fixture_run",
        "metrics": {"accuracy": 0.78, "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1},
        "training_majority_baseline": {
            "metrics": {"accuracy": 0.65, "balanced_accuracy": 0.5, "macro_f1": baseline_macro_f1},
        },
        "model_quality_approval": "NOT_ESTABLISHED",
        "approved_for_delivery": False,
    }


@pytest.fixture
def paths(tmp_path):
    evaluation = tmp_path / "evaluation_report.json"
    evaluation.write_text(json.dumps(evaluation_report()), encoding="utf-8")
    return evaluation, tmp_path / "acceptance_report.json", tmp_path / "metrics.json"


def write(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


def test_project_criteria_are_valid_team_criteria():
    criteria, _ = gate.load_criteria(CRITERIA_FILE)
    assert criteria["status"] == "team_defined"


def test_passing_report_records_checks_without_approval(paths):
    evaluation, output, metrics = paths
    report = gate.accept_run(evaluation, CRITERIA_FILE, output, metrics)
    assert report["gate_status"] == "passed"
    assert [check["name"] for check in report["checks"]] == [
        "macro_f1", "balanced_accuracy", "macro_f1_gain_over_training_majority"]
    assert all(check["passed"] for check in report["checks"])
    assert report["checks"][2]["observed"] == pytest.approx(0.34)
    assert report["model_quality_approval"] == "TEAM_CRITERIA_MET"
    assert report["approved_for_delivery"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report
    summary = json.loads(metrics.read_text(encoding="utf-8"))
    assert summary["macro_f1"] == 0.74
    assert summary["training_majority_macro_f1"] == 0.40
    assert summary["gate_passed"] is True


@pytest.mark.parametrize("values, failed", [
    ({"macro_f1": 0.60, "baseline_macro_f1": 0.30}, "macro_f1"),
    ({"balanced_accuracy": 0.60}, "balanced_accuracy"),
    ({"macro_f1": 0.70, "baseline_macro_f1": 0.55}, "macro_f1_gain_over_training_majority"),
])
def test_each_criterion_can_fail_the_gate(paths, values, failed):
    evaluation, output, metrics = paths
    write(evaluation, evaluation_report(**values))
    report = gate.accept_run(evaluation, CRITERIA_FILE, output, metrics)
    assert report["gate_status"] == "failed"
    assert [check["name"] for check in report["checks"] if not check["passed"]] == [failed]
    assert report["model_quality_approval"] == "NOT_ESTABLISHED"
    assert report["approved_for_delivery"] is False
    assert json.loads(metrics.read_text(encoding="utf-8"))["gate_passed"] is False


def test_threshold_equal_to_observed_value_passes(paths, tmp_path):
    evaluation, output, _ = paths
    criteria = json.loads(CRITERIA_FILE.read_text(encoding="utf-8"))
    criteria.update(min_macro_f1=0.74, min_balanced_accuracy=0.73)
    write(tmp_path / "criteria.json", criteria)
    assert gate.accept_run(evaluation, tmp_path / "criteria.json", output)["gate_status"] == "passed"


@pytest.mark.parametrize("macro_f1, baseline_macro_f1, gate_status", [
    (0.68, 0.49, "failed"),           # gain 0.19: below the minimum
    (0.69, 0.49, "passed"),           # gain 0.19999999999999996: the minimum after rounding
    (0.70, 0.49, "passed"),           # gain 0.21: above the minimum
    (0.6899999, 0.49, "failed"),      # gain 0.1999999: a real shortfall is not tolerated
])
def test_gain_boundary_tolerates_only_floating_point_rounding(paths, macro_f1, baseline_macro_f1, gate_status):
    evaluation, output, _ = paths
    write(evaluation, evaluation_report(macro_f1=macro_f1, baseline_macro_f1=baseline_macro_f1))
    report = gate.accept_run(evaluation, CRITERIA_FILE, output)
    gain = report["checks"][2]
    assert gain["name"] == "macro_f1_gain_over_training_majority"
    assert gain["minimum"] == 0.2
    assert report["gate_status"] == gate_status
    assert report["comparison_tolerance"] == gate.COMPARISON_TOLERANCE == 1e-9


@pytest.mark.parametrize("change", [
    {"schema_name": "other"},
    {"evaluation_status": "failed"},
    {"approved_for_delivery": True},
    {"metrics": {"accuracy": 0.7, "macro_f1": 0.7}},
    {"metrics": {"accuracy": 0.7, "balanced_accuracy": 0.7, "macro_f1": float("nan")}},
    {"training_majority_baseline": {}},
])
def test_unexpected_evaluation_reports_are_rejected_without_output(paths, change):
    evaluation, output, metrics = paths
    write(evaluation, {**evaluation_report(), **change})
    with pytest.raises(ValueError):
        gate.accept_run(evaluation, CRITERIA_FILE, output, metrics)
    assert not output.exists() and not metrics.exists()


@pytest.mark.parametrize("change", [
    {"status": "proposed"},
    {"min_macro_f1": 1.5},
    {"min_balanced_accuracy": "0.6"},
    {"min_recall": 0.5},
])
def test_invalid_criteria_are_rejected(paths, tmp_path, change):
    evaluation, output, _ = paths
    criteria = {**json.loads(CRITERIA_FILE.read_text(encoding="utf-8")), **change}
    write(tmp_path / "criteria.json", criteria)
    with pytest.raises(ValueError):
        gate.accept_run(evaluation, tmp_path / "criteria.json", output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(paths):
    evaluation, output, _ = paths
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.accept_run(evaluation, CRITERIA_FILE, output)
    assert output.read_text(encoding="utf-8") == "previous"


@pytest.mark.parametrize("values, code", [({}, 0), ({"balanced_accuracy": 0.5}, 1)])
def test_cli_exit_code_reflects_the_gate(paths, values, code):
    evaluation, output, _ = paths
    write(evaluation, evaluation_report(**values))
    command = [sys.executable, "-B", gate.__file__, "--evaluation", str(evaluation),
               "--criteria", str(CRITERIA_FILE), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == code, result.stderr
    assert output.exists()
