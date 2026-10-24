"""Evaluate a technically validated reference run without training or serving."""

import argparse
import hashlib
import io
import json
import logging
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from reference_contract import CONTRACT_VERSION, FEATURES


def assert_local_path(path, *, regular_file=False):
    """Reject symbolic links and Windows reparse points before reading a path."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked paths are not supported: {component}")
    if regular_file and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    return path


def identity(content):
    return {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def target_values(frame, name):
    if frame.empty or frame.shape[1] != 1:
        raise ValueError(f"{name} must contain one nonempty target column.")
    values = frame.to_numpy().ravel()
    if not np.issubdtype(values.dtype, np.number) or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary codes 0 and 1.")
    return values


def metrics(targets, predictions):
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=[0, 1], average="macro", zero_division=0)),
    }


def evaluate_run(run_directory, output_path):
    """Write one new descriptive report from trusted local run artifacts.

    Input hashes identify the observed files; they do not establish the original
    training lineage or independently prove how the held-out split was made.
    """
    run_directory = assert_local_path(run_directory)
    if not run_directory.is_dir():
        raise ValueError("The run directory must be an existing directory.")
    output_path = Path(output_path).absolute()
    assert_local_path(output_path.parent)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")

    inputs = {}

    def snapshot(relative):
        path = assert_local_path(run_directory / relative, regular_file=True)
        content = path.read_bytes()
        inputs[relative] = {"path": str(path), **identity(content)}
        return content

    validation = json.loads(snapshot("validation_report.json"))
    if (
        validation.get("scope") != "technical_data_and_model_validation"
        or validation.get("validation_status") != "passed"
        or validation.get("contract_version") != CONTRACT_VERSION
        or validation.get("run_id") != run_directory.name
        or validation.get("approved_for_delivery") is not False
    ):
        raise ValueError("The run does not have a matching passed technical validation report.")

    features = pd.read_csv(io.BytesIO(snapshot("preprocessed/X_test.csv")))
    train_targets = target_values(pd.read_csv(io.BytesIO(snapshot("preprocessed/y_train.csv"))), "y_train")
    test_targets = target_values(pd.read_csv(io.BytesIO(snapshot("preprocessed/y_test.csv"))), "y_test")
    if features.empty or tuple(features.columns) != FEATURES or len(features) != len(test_targets):
        raise ValueError("Held-out features and targets must match the prepared feature contract and row count.")
    values = features.to_numpy()
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Held-out features must be finite numeric values.")
    # Both classes are needed for the agreed binary comparison metrics.
    if set(test_targets) != {0, 1}:
        raise ValueError("Held-out targets must include both binary classes for this evaluation.")

    model_bytes = snapshot("model/trained_model.joblib")
    model_identity = inputs["model/trained_model.joblib"]
    if validation.get("model_sha256") != model_identity["sha256"]:
        raise ValueError("Model bytes do not match the technical validation report.")
    # Deserialize only the exact checked snapshot, never a second model read.
    model = joblib.load(io.BytesIO(model_bytes))
    del model_bytes
    if (
        not isinstance(model, RandomForestClassifier)
        or tuple(getattr(model, "feature_names_in_", ())) != FEATURES
        or getattr(model, "n_features_in_", None) != len(FEATURES)
        or tuple(getattr(model, "classes_", ())) != (0, 1)
    ):
        raise ValueError("Model estimator, prepared features or binary classes differ from the reference contract.")
    predictions = np.asarray(model.predict(features))
    if predictions.shape != test_targets.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Model predictions must be one binary class code per held-out row.")

    counts = [int(np.count_nonzero(train_targets == code)) for code in (0, 1)]
    majority_code = int(np.argmax(counts))
    report = {
        "schema_name": "reference_run_evaluation_v1",
        "scope": "descriptive_prepared_holdout_evaluation",
        "evaluation_status": "completed",
        "run_id": run_directory.name,
        "run_directory": str(run_directory),
        "contract_version": CONTRACT_VERSION,
        "test_rows": len(test_targets),
        "feature_count": len(FEATURES),
        "inputs": inputs,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "metrics": metrics(test_targets, predictions),
        "training_majority_baseline": {
            "class_code": majority_code,
            "selection_basis": "training_targets_only",
            "tie_break": "lowest_class_code",
            "training_class_counts": {"0": counts[0], "1": counts[1]},
            "metrics": metrics(test_targets, np.full(test_targets.shape, majority_code)),
        },
        "quality_thresholds_enforced": False,
        "model_quality_approval": "NOT_ESTABLISHED",
        "approved_for_delivery": False,
        "limitations": [
            "Observed file hashes do not independently prove training or split lineage.",
            "Inherited severity-dependent selection and prediction-time feature limits remain.",
            "Binary class codes carry no newly inferred domain labels.",
        ],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # A concurrent input change must not produce an apparently stable report.
    for recorded in inputs.values():
        path = assert_local_path(recorded["path"], regular_file=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
        if digest != recorded["sha256"] or size != recorded["size_bytes"]:
            raise ValueError(f"Evaluation input changed during execution: {path}")
    assert_local_path(output_path.parent)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = evaluate_run(args.run_dir, args.output)
    except Exception:
        logging.exception("Reference run evaluation failed.")
        return 1
    print(f"Reference run evaluation written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
