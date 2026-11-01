"""Run one reference pipeline stage in a shared run directory.

The stages are prepare, train and evaluate. `dvc.yaml` and
`compose.pipeline.yaml` call them in this order; each stage reads only the
outputs of the previous one and creates new files without overwriting.
Stages do not record MLflow runs; tracked runs use `run_pipeline.py`.
"""

import argparse
import json
import logging
from pathlib import Path

from accept_run import accept_run
from data.make_dataset import RAW_FILENAMES, process_data
from evaluate_run import evaluate_run
from models.train_model import train_model
from run_pipeline import validate_model_artifact, validate_prepared_data, write_validation_report


def prepare(raw_directory, run_directory):
    raw_directory = Path(raw_directory).resolve(strict=True)
    run_directory = Path(run_directory).resolve()
    if run_directory == raw_directory or raw_directory in run_directory.parents:
        raise ValueError("The run directory must be outside the raw directory.")
    processed_directory = run_directory / "preprocessed"
    process_data(*(raw_directory / name for name in RAW_FILENAMES), processed_directory)
    data_validation = validate_prepared_data(processed_directory)
    with (run_directory / "data_validation.json").open("x", encoding="utf-8") as stream:
        json.dump(data_validation, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return processed_directory


def train(run_directory):
    run_directory = Path(run_directory).resolve(strict=True)
    data_validation = json.loads((run_directory / "data_validation.json").read_text(encoding="utf-8"))
    model_path = train_model(
        run_directory / "preprocessed",
        run_directory / "model" / "trained_model.joblib",
    )
    model_validation = validate_model_artifact(model_path)
    return write_validation_report(
        run_directory / "validation_report.json", run_directory, data_validation, model_validation,
    )


def evaluate(run_directory, criteria_path):
    run_directory = Path(run_directory).resolve(strict=True)
    evaluation_path = evaluate_run(run_directory, run_directory / "evaluation_report.json")
    return accept_run(
        evaluation_path,
        criteria_path,
        run_directory / "acceptance_report.json",
        run_directory / "metrics.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = stages.add_parser("prepare")
    prepare_parser.add_argument("--raw-dir", required=True, type=Path)
    prepare_parser.add_argument("--run-dir", required=True, type=Path)
    train_parser = stages.add_parser("train")
    train_parser.add_argument("--run-dir", required=True, type=Path)
    evaluate_parser = stages.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", required=True, type=Path)
    evaluate_parser.add_argument("--criteria", required=True, type=Path)
    args = parser.parse_args()

    try:
        if args.stage == "prepare":
            result = prepare(args.raw_dir, args.run_dir)
        elif args.stage == "train":
            result = train(args.run_dir)
        else:
            report = evaluate(args.run_dir, args.criteria)
            print(f"Proposed quality gate {report['gate_status']}.")
            # A failed gate stops `dvc repro` and the dependent Compose services.
            return 0 if report["gate_status"] == "passed" else 1
    except Exception:
        logging.exception("Pipeline stage %s failed.", args.stage)
        return 2
    print(f"Pipeline stage {args.stage} completed: {result}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
