"""Record one finished stage-pipeline run (prepare, train, evaluate) in MLflow.

The recorder does not run stages. It is called after evaluate by the
orchestrator that started them: the Airflow DAG `reference_pipeline` (task
`track`) or, later, a `dvc repro` run. Both use the same experiment, tags and
checks, so a run can be found by its orchestrator run ID:

    tags.orchestrator_run_id = '<Airflow run ID>'

The gate is checked with the same function as staged runs (`validate_gate` in
track_staged_run.py). Raw data, prepared data and model are recorded with
SHA-256 and with the MD5 hash that DVC writes into dvc.lock, so the MLflow run
names the exact data and model versions produced by that run.

Exit codes: 0 when the run was recorded (FINISHED or FAILED), 2 when it could
not be recorded. A failed gate is recorded as FAILED, not as an error here.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.reference_contract import CONTRACT_VERSION
from src.accept_run import load_criteria
from src.tracking.import_reference_run import ImportFailure, LocalClient
from src.tracking.track_pipeline_run import TrackingFailure, file_identity, json_bytes, read_json, require, write_exclusive
from src.tracking.track_staged_run import OUTPUTS, RAW_FILES, validate_gate

EXPERIMENT = "car_accident_stage_runs"
ORCHESTRATORS = ("airflow", "dvc")
RESULT_FILE = "mlflow_tracking.json"
VERSIONS_FILE = "run_versions.json"
# First output of each stage; a missing file means that stage did not complete.
STAGE_OUTPUTS = (
    ("prepare", "data_validation.json"),
    ("train", "validation_report.json"),
    ("evaluate", "acceptance_report.json"),
)
# Same paths as the `outs` of dvc.yaml, relative to the run directory.
VERSIONED_OUTPUTS = {"prepared_data": "preprocessed", "model": "model/trained_model.joblib"}
LOCAL_SERVER_VERSION = "3.17.0"


def md5_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def dvc_hash(path):
    """MD5 as DVC 3 writes it into dvc.lock: file MD5, or `<md5>.dir` for a directory."""
    path = Path(path)
    if path.is_file():
        return {"md5": md5_file(path), "size": path.stat().st_size}
    require(path.is_dir(), f"Missing versioned output {path.name}.")
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix())
    require(files and not any(item.is_symlink() for item in path.rglob("*")), "Versioned directory is empty or linked.")
    entries = [{"md5": md5_file(item), "relpath": item.relative_to(path).as_posix()} for item in files]
    digest = hashlib.md5(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
    return {"md5": digest + ".dir", "size": sum(item.stat().st_size for item in files), "nfiles": len(files)}


def stage_outcome(directory, gate):
    """Return (successful, failed stage) from the files the stages left behind."""
    for stage, name in STAGE_OUTPUTS:
        if not (directory / name).is_file():
            return False, stage
    if gate is None or gate["gate_status"] != "passed":
        return False, "evaluate"
    return all((directory / name).is_file() for name in OUTPUTS), None


def version_record(directory, raw):
    record = {"schema_name": "stage_run_versions_v1", "hash_scope": "sha256 of bytes; md5 as written by DVC 3 in dvc.lock",
              "raw_data": {"dvc": dvc_hash(raw), "files": {name: file_identity(raw / name) for name in RAW_FILES}}}
    for key, relative in VERSIONED_OUTPUTS.items():
        path = directory / relative
        if not path.exists():
            continue
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        record[key] = {"path": relative, "dvc": dvc_hash(path),
                       "files": {item.relative_to(directory).as_posix(): file_identity(item) for item in files}}
    return record


def parameters(criteria, criteria_path, directory, extra):
    values = {"contract_version": CONTRACT_VERSION, "criteria_path": str(criteria_path),
              "criteria_version": criteria["criteria_version"]}
    values.update({key: str(value) for key, value in criteria.items() if key.startswith("min_")})
    report = directory / "validation_report.json"
    if report.is_file():
        configuration = read_json(report.read_bytes()).get("model_validation", {}).get("estimator_configuration", {})
        if configuration:
            values["estimator_type"] = configuration["estimator_type"]
            values.update({"model_" + key: json.dumps(value) for key, value in configuration["parameters"].items()})
    values.update(extra)
    return values


def record_stage_run(run_directory, raw_directory, criteria_path, orchestrator, orchestrator_run_id, source_commit,
                     uri, *, extra_parameters=None, client=None):
    require(orchestrator in ORCHESTRATORS, "Unknown orchestrator.")
    require(bool(orchestrator_run_id) and len(orchestrator_run_id) <= 250, "Missing orchestrator run ID.")
    require(re.fullmatch(r"[0-9a-f]{40}|unknown", source_commit) is not None, "Expected a 40-character commit or 'unknown'.")
    directory, raw = Path(run_directory).resolve(strict=True), Path(raw_directory).resolve(strict=True)
    result_path = directory / RESULT_FILE
    require(not result_path.exists(), "Run was already recorded; no duplicate or resume.")
    criteria_path = Path(criteria_path)
    criteria, _ = load_criteria(criteria_path)
    extra = dict(extra_parameters or {})
    record_path = directory / "airflow_run.json"
    if orchestrator == "airflow":
        record = read_json(record_path.read_bytes())
        require(record["airflow_run_id"] == orchestrator_run_id and record["criteria"] == str(criteria_path)
                and record["source_commit"] == source_commit, "airflow_run.json differs from this run.")
        extra.setdefault("quality_gate", record["quality_gate"])

    # Checks the gate report against the criteria and the current evaluation inputs.
    # Success is decided afterwards from the files, so no success is asserted here.
    gate, metrics = validate_gate(directory, criteria_path, False)
    successful, failed_stage = stage_outcome(directory, gate)
    if gate is not None:
        metrics["macro_f1_gain_over_training_majority"] = next(
            check["observed"] for check in gate["checks"] if check["name"] == "macro_f1_gain_over_training_majority")
    versions = version_record(directory, raw)
    versions_bytes = json_bytes(versions)

    artifacts = {"outputs/" + name: directory / name for name in OUTPUTS if (directory / name).is_file()}
    if record_path.is_file():
        artifacts["evidence/airflow_run.json"] = record_path
    identities = {name: file_identity(path) for name, path in artifacts.items()}

    terminal = "FINISHED" if successful else "FAILED"
    gate_status = gate["gate_status"] if gate else "not_established"
    tags = {"orchestrator": orchestrator, "orchestrator_run_id": orchestrator_run_id,
            "pipeline_run_id": directory.name, "source_commit": source_commit,
            "tracking_scope": "stage_pipeline", "gate_status": gate_status,
            "model_quality_approval": gate["model_quality_approval"] if gate else "NOT_ESTABLISHED",
            "failed_stage": failed_stage or "none", "approved_for_delivery": "false",
            "criteria_sha256": file_identity(criteria_path)["sha256"],
            "raw_data_dvc_md5": versions["raw_data"]["dvc"]["md5"]}
    for key in VERSIONED_OUTPUTS:
        if key in versions:
            tags[key + "_dvc_md5"] = versions[key]["dvc"]["md5"]
    if "model" in versions:
        tags["model_sha256"] = versions["model"]["files"][VERSIONED_OUTPUTS["model"]]["sha256"]
    if orchestrator == "airflow":
        tags["airflow_run_id"] = orchestrator_run_id
    params = parameters(criteria, criteria_path, directory, extra)

    state = {"schema_name": "stage_run_tracking_v1", "orchestrator": orchestrator, "orchestrator_run_id": orchestrator_run_id,
             "mlflow_run_id": None, "mlflow_experiment": EXPERIMENT, "remote_status": "not_created", "status": "preparing",
             "gate_status": gate_status, "failed_stage": failed_stage, "approved_for_delivery": False}
    client = client or LocalClient(uri)
    known_run = None
    try:
        version = client.server_version()
        require(version == LOCAL_SERVER_VERSION if client.target == "local" else re.fullmatch(r"3\.\d+\.\d+", version) is not None,
                "Unsupported tracking server version.")
        tags.update(tracking_target=client.target, tracking_server_version=version)
        state["tracking_target"] = client.target
        experiments = client.post("experiments/search", {"filter": f"name = '{EXPERIMENT}'", "view_type": 3, "max_results": 2}).get("experiments", [])
        require(len(experiments) <= 1 and (not experiments or experiments[0]["lifecycle_stage"] == "active"), "Ambiguous or deleted experiment.")
        experiment_id = experiments[0]["experiment_id"] if experiments else client.post("experiments/create", {"name": EXPERIMENT})["experiment_id"]
        quoted = orchestrator_run_id.replace("'", "\\'")
        duplicates = client.post("runs/search", {"experiment_ids": [experiment_id], "run_view_type": 3, "max_results": 1,
                                 "filter": f"tags.orchestrator_run_id = '{quoted}' and tags.orchestrator = '{orchestrator}'"}).get("runs", [])
        require(not duplicates, "This orchestrator run is already in MLflow; no duplicate or resume.")
        state.update(status="unknown", remote_status="unknown")
        start = int(datetime.now(timezone.utc).timestamp() * 1000)
        created = client.post("runs/create", {"experiment_id": experiment_id, "run_name": directory.name, "start_time": start,
                                              "tags": [{"key": k, "value": v} for k, v in tags.items()]})["run"]
        observed_tags = {v["key"]: v["value"] for v in created["data"].get("tags", [])}
        require(re.fullmatch(r"[0-9a-f]{32}", created["info"]["run_id"]) is not None
                and created["info"]["experiment_id"] == experiment_id and created["info"]["status"] == "RUNNING"
                and all(observed_tags.get(k) == v for k, v in tags.items()), "Unconfirmed run creation; preserve unknown remote state.")
        known_run = created
        state.update(mlflow_run_id=created["info"]["run_id"], remote_status="RUNNING", status="recording")
        evidence = directory / "mlflow_evidence"
        evidence.mkdir()
        write_exclusive(evidence / VERSIONS_FILE, versions_bytes)
        artifacts["evidence/" + VERSIONS_FILE] = evidence / VERSIONS_FILE
        identities["evidence/" + VERSIONS_FILE] = file_identity(evidence / VERSIONS_FILE)
        for name, path in artifacts.items():
            require(file_identity(path) == identities[name], "Artifact changed before upload.")
            client.upload(known_run, name, path, identities[name]["sha256"])
            require(client.download_hash(known_run, name, identities[name]["size_bytes"]) == identities[name]["sha256"],
                    "Artifact round trip differs.")
        require(version_record(directory, raw) == versions, "Data or model changed during recording.")
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        client.post("runs/log-batch", {"run_id": state["mlflow_run_id"],
                    "params": [{"key": k, "value": v} for k, v in params.items()],
                    "metrics": [{"key": k, "value": v, "timestamp": timestamp, "step": 0} for k, v in metrics.items()]})
        client.post("runs/update", {"run_id": state["mlflow_run_id"], "status": terminal, "end_time": timestamp})
        observed = client.run(state["mlflow_run_id"])
        actual_tags = {v["key"]: v["value"] for v in observed["data"].get("tags", [])}
        actual_params = {v["key"]: v["value"] for v in observed["data"].get("params", [])}
        actual_metrics = {v["key"]: v["value"] for v in observed["data"].get("metrics", [])}
        require(observed["info"]["status"] == terminal and observed["info"]["lifecycle_stage"] == "active"
                and all(actual_tags.get(k) == v for k, v in tags.items())
                and all(actual_params.get(k) == v for k, v in params.items())
                and all(actual_metrics.get(k) == v for k, v in metrics.items()), "Tracking completion is not confirmed.")
        state.update(status="recorded", remote_status=terminal)
        write_exclusive(result_path, json_bytes(state))
        return state
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, (TrackingFailure, ImportFailure)):
            state["guard_message"] = str(error)
        if known_run is not None:
            state["remote_status"] = "unknown"
            try:
                end = int(datetime.now(timezone.utc).timestamp() * 1000)
                client.post("runs/update", {"run_id": state["mlflow_run_id"], "status": "FAILED", "end_time": end})
                if client.run(state["mlflow_run_id"])["info"]["status"] == "FAILED":
                    state["remote_status"] = "FAILED"
            except Exception as secondary:
                state["secondary_error_type"] = type(secondary).__name__
        if not result_path.exists():
            write_exclusive(result_path, json_bytes(state))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path, help="Criteria path exactly as passed to evaluate.")
    parser.add_argument("--orchestrator", required=True, choices=ORCHESTRATORS)
    parser.add_argument("--orchestrator-run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="Extra MLflow parameter.")
    args = parser.parse_args()
    extra = dict(item.split("=", 1) for item in args.param)
    try:
        result = record_stage_run(args.run_dir, args.raw_dir, args.criteria, args.orchestrator, args.orchestrator_run_id,
                                  args.source_commit, args.tracking_uri, extra_parameters=extra)
    except Exception as error:
        detail = ": " + str(error) if isinstance(error, (TrackingFailure, ImportFailure)) else ""
        print("STAGE_TRACKING_FAILED: " + type(error).__name__ + detail)
        return 2
    print(f"MLflow run {result['mlflow_run_id']} in {EXPERIMENT} ({result['tracking_target']}): {result['remote_status']}, "
          f"gate {result['gate_status']}, orchestrator run {result['orchestrator_run_id']}.")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
