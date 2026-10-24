"""Run reference preprocessing and training in a new, isolated directory.

The inherited target mapping and severity-based row selection are preserved.
The estimator has no random seed; repeated fits may produce different results.
A completed workflow includes technical validation, descriptive evaluation and
enabled tracking. Completion does not confer model-quality or delivery approval.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data.make_dataset import RAW_FILENAMES, process_data
from evaluate_run import evaluate_run
from models.train_model import train_model
from reference_contract import CONTRACT_VERSION, FEATURES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_prepared_data(directory):
    """Check both prepared partitions without fitting or evaluating a model."""
    partitions = {}
    for partition in ("train", "test"):
        features = pd.read_csv(directory / f"X_{partition}.csv")
        targets = pd.read_csv(directory / f"y_{partition}.csv")
        if features.empty or targets.empty or targets.shape[1] != 1:
            raise ValueError(f"Prepared {partition} features or target are empty or invalid.")
        if len(features) != len(targets):
            raise ValueError(f"Prepared {partition} feature and target row counts differ.")
        if tuple(features.columns) != FEATURES:
            raise ValueError(f"Prepared {partition} feature contract differs.")
        feature_values = features.to_numpy()
        if not np.issubdtype(feature_values.dtype, np.number) or not np.isfinite(feature_values).all():
            raise ValueError(f"Prepared {partition} features must be finite numeric values.")
        target_values = targets.to_numpy()
        if not np.issubdtype(target_values.dtype, np.number) or not np.isin(target_values, (0, 1)).all():
            raise ValueError(f"Prepared {partition} target codes must be 0 or 1.")
        partitions[partition] = {
            "row_count": len(features),
            "feature_count": len(FEATURES),
            "checks": {
                "non_empty": "passed",
                "matching_rows": "passed",
                "feature_contract": "passed",
                "finite_numeric_features": "passed",
                "target_codes": "passed",
            },
        }
    return {"checked_at_utc": utc_now(), "partitions": partitions}


def validate_model_artifact(model_path):
    """Check the newly produced model contract; never call inference or scoring."""
    with Path(model_path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        model = joblib.load(stream)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Produced artifact is not a Random Forest model.")
    if tuple(getattr(model, "feature_names_in_", ())) != FEATURES or getattr(model, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Produced model feature contract differs.")
    if tuple(getattr(model, "classes_", ())) != (0, 1):
        raise ValueError("Produced model class codes must match [0, 1].")
    return {
        "checked_at_utc": utc_now(),
        "model_path": str(model_path),
        "model_sha256": digest,
        "checks": {
            "estimator_type": "passed",
            "feature_contract": "passed",
            "class_codes": "passed",
        },
    }


def run_pipeline(raw_directory, run_directory, tracker=None):
    raw_directory = Path(raw_directory).resolve(strict=True)
    run_directory = Path(run_directory).resolve()
    raw_paths = [raw_directory / name for name in RAW_FILENAMES]

    for path in raw_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required input is missing or empty: {path}")

    if run_directory == raw_directory or raw_directory in run_directory.parents:
        raise ValueError("The run directory must be outside the raw directory.")

    # Claim a new directory; never resume or overwrite an existing run.
    run_directory.mkdir(parents=True, exist_ok=False)
    status_path = run_directory / "run_status.json"
    record = {
        "status": "running",
        "started_at_utc": utc_now(),
        "raw_directory": str(raw_directory),
        "run_directory": str(run_directory),
        "random_state": None,
        "technical_validation_status": "pending",
        "evaluation_status": "pending",
        "model_quality_approval": "NOT_ESTABLISHED",
        "approved_for_delivery": False,
    }

    def write_status():
        if tracker is not None:
            record["tracking"] = dict(tracker.state)
        status_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

    def phase(name):
        record["phase"] = name
        if tracker is not None:
            tracker.phase(name)
        write_status()

    try:
        write_status()
        if tracker is not None:
            tracker.start(raw_directory, run_directory, record, FEATURES, CONTRACT_VERSION)
            write_status()
        processed_directory = run_directory / "preprocessed"
        phase("preprocessing")
        process_data(*raw_paths, processed_directory)
        phase("data_validation")
        data_validation = validate_prepared_data(processed_directory)
        phase("training")
        model_path = train_model(
            processed_directory,
            run_directory / "model" / "trained_model.joblib",
        )
        phase("model_validation")
        model_validation = validate_model_artifact(model_path)
        phase("report_writing")
        report_path = run_directory / "validation_report.json"
        report = {
            "scope": "technical_data_and_model_validation",
            "validation_status": "passed",
            "contract_version": CONTRACT_VERSION,
            "run_id": run_directory.name,
            "run_directory": str(run_directory),
            "model_path": model_validation["model_path"],
            "model_sha256": model_validation["model_sha256"],
            "data_validation": data_validation,
            "model_validation": model_validation,
            "checked_at_utc": utc_now(),
            "approved_for_delivery": False,
        }
        # Only this new run receives a technical report; historical reports stay unchanged.
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        record.update(
            technical_validation_status="passed",
            validation_report_path=str(report_path),
            model_path=str(model_path),
        )
        phase("evaluation")
        record["evaluation_status"] = "running"
        write_status()
        evaluation_path = evaluate_run(run_directory, run_directory / "evaluation_report.json")
        record.update(
            evaluation_status="completed",
            evaluation_report_path=str(evaluation_path),
        )
        write_status()
        if tracker is not None:
            phase("artifact_recording")
            tracker.finish(report_path, model_path, evaluation_path)
            tracker.state["phase"] = "local_completion"
        record.update(
            status="completed",
            finished_at_utc=utc_now(),
            model_path=str(model_path),
            validation_report_path=str(report_path),
        )
        write_status()
    except Exception as error:
        if record["evaluation_status"] == "running":
            record["evaluation_status"] = "failed"
        if tracker is not None:
            try:
                tracker.fail(error)
            except Exception as tracking_error:
                tracker.state.update(status="unknown", secondary_error_type=type(tracking_error).__name__)
                logging.exception("Could not record the tracking failure.")
        record.update(
            status="failed",
            finished_at_utc=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        try:
            write_status()
        except OSError:
            logging.exception("Could not persist the failed run status.")
        raise

    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--tracking-context", type=Path)
    args = parser.parse_args()
    if args.tracking_uri is not None and not args.tracking_uri:
        parser.error("--tracking-uri must not be empty")
    if (args.tracking_uri is None) != (args.tracking_context is None):
        parser.error("--tracking-uri and --tracking-context must be supplied together")

    try:
        if args.tracking_uri is not None:
            from tracking.track_pipeline_run import PipelineTracker

            tracker = PipelineTracker(args.tracking_uri, args.tracking_context)
            result = run_pipeline(args.raw_dir, args.run_dir, tracker=tracker)
        else:
            result = run_pipeline(args.raw_dir, args.run_dir)
    except Exception:
        logging.exception("Reference pipeline failed.")
        return 1

    print(f"Reference pipeline completed: {result}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
