"""Check a descriptive evaluation report against proposed quality criteria.

The criteria are a team proposal and have not been agreed with the supervisor.
A passed gate records that the proposal is met; it does not establish model
quality approval or delivery approval.
"""

import argparse
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path


METRIC_NAMES = {"accuracy", "balanced_accuracy", "macro_f1"}
CRITERIA_KEYS = {
    "criteria_version", "status", "min_macro_f1", "min_balanced_accuracy",
    "min_macro_f1_gain_over_training_majority",
}


def read_json(path):
    content = Path(path).read_bytes()
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def unit_interval(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")
    return float(value)


def load_criteria(path):
    criteria, source = read_json(path)
    if not isinstance(criteria, dict) or set(criteria) != CRITERIA_KEYS:
        raise ValueError("Acceptance criteria keys differ from the expected selection.")
    if criteria["status"] != "proposed" or not isinstance(criteria["criteria_version"], str):
        raise ValueError("Acceptance criteria must be a versioned proposal.")
    for key in CRITERIA_KEYS - {"criteria_version", "status"}:
        unit_interval(criteria[key], key)
    return criteria, source


def evaluation_metrics(report, name):
    values = report.get(name) if name == "metrics" else report.get(name, {}).get("metrics")
    if not isinstance(values, dict) or set(values) != METRIC_NAMES:
        raise ValueError(f"Evaluation {name} metrics differ from the expected selection.")
    return {key: unit_interval(value, key) for key, value in values.items()}


def accept_run(evaluation_path, criteria_path, output_path, metrics_path=None):
    """Write one new gate report and return it; never overwrite existing files."""
    evaluation, evaluation_source = read_json(evaluation_path)
    if (
        evaluation.get("schema_name") != "reference_run_evaluation_v1"
        or evaluation.get("evaluation_status") != "completed"
        or evaluation.get("approved_for_delivery") is not False
    ):
        raise ValueError("The evaluation report is not a completed reference evaluation.")
    model = evaluation_metrics(evaluation, "metrics")
    baseline = evaluation_metrics(evaluation, "training_majority_baseline")
    criteria, criteria_source = load_criteria(criteria_path)

    checks = [
        ("macro_f1", model["macro_f1"], criteria["min_macro_f1"]),
        ("balanced_accuracy", model["balanced_accuracy"], criteria["min_balanced_accuracy"]),
        ("macro_f1_gain_over_training_majority", model["macro_f1"] - baseline["macro_f1"],
         criteria["min_macro_f1_gain_over_training_majority"]),
    ]
    results = [
        {"name": name, "observed": observed, "minimum": minimum, "passed": observed >= minimum}
        for name, observed, minimum in checks
    ]
    passed = all(result["passed"] for result in results)
    report = {
        "schema_name": "reference_run_acceptance_v1",
        "scope": "proposed_model_quality_gate",
        "run_id": evaluation.get("run_id"),
        "evaluation_report": evaluation_source,
        "criteria": {"source": criteria_source, "version": criteria["criteria_version"],
                     "status": criteria["status"]},
        "checks": results,
        "gate_status": "passed" if passed else "failed",
        "model_quality_approval": "NOT_ESTABLISHED",
        "approved_for_delivery": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with Path(output_path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if metrics_path is not None:
        # Flat values for `dvc metrics show` and `dvc metrics diff`.
        summary = {**model, **{f"training_majority_{key}": value for key, value in baseline.items()},
                   "gate_passed": passed}
        with Path(metrics_path).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    try:
        report = accept_run(args.evaluation, args.criteria, args.output, args.metrics)
    except Exception:
        logging.exception("Acceptance check failed to run.")
        return 2
    print(f"Proposed quality gate {report['gate_status']}: {args.output}")
    return 0 if report["gate_status"] == "passed" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
