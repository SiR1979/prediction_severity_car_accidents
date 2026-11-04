"""Record a finished three-container workflow without running pipeline stages.

Recording is deliberately performed after stage execution. It is not live stage
monitoring. Partial outputs and failed gates are retained as failed workflows.
"""

import argparse
import base64
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.accept_run import COMPARISON_TOLERANCE, evaluation_metrics, load_criteria
from src.reference_contract import CONTRACT_VERSION
from src.tracking.import_reference_run import ImportFailure, LocalClient
from src.tracking.track_pipeline_run import TrackingFailure, file_identity, json_bytes, read_json, require, sha256, write_exclusive


SOURCES = (
    "src/pipeline_stages.py", "src/run_pipeline.py", "src/accept_run.py",
    "src/evaluate_run.py", "src/reference_contract.py", "src/data/make_dataset.py",
    "src/models/train_model.py", "src/tracking/import_reference_run.py",
    "src/tracking/track_pipeline_run.py", "src/tracking/track_staged_run.py",
)
CONFIGURATIONS = (
    "Dockerfile", "requirements-runtime.txt", "src/acceptance_criteria.json",
    "scripts/run_staged_tracked_pipeline.ps1",
)
RAW_FILES = ("usagers-2021.csv", "caracteristiques-2021.csv", "lieux-2021.csv", "vehicules-2021.csv")
STAGES = ("prepare", "train", "evaluate")
OUTPUTS = (
    "data_validation.json", "validation_report.json", "evaluation_report.json",
    "acceptance_report.json", "metrics.json", "model/trained_model.joblib",
    "preprocessed/X_train.csv", "preprocessed/X_test.csv",
    "preprocessed/y_train.csv", "preprocessed/y_test.csv",
)


def milliseconds(value):
    observed = datetime.fromisoformat(value)
    require(observed.utcoffset() is not None and observed.utcoffset().total_seconds() == 0, "Expected UTC timestamp.")
    return int(observed.timestamp() * 1000)


def validate_receipt(context, receipt):
    require(receipt["schema_name"] == "staged_pipeline_receipt_v1", "Unexpected receipt schema.")
    require(receipt["invocation_id"] == context["invocation_id"], "Receipt belongs to another invocation.")
    stages = receipt["stages"]
    require(isinstance(stages, list) and 1 <= len(stages) <= 3, "Expected an attempted stage prefix.")
    previous_end = milliseconds(context["observed_at_utc"])
    for index, stage in enumerate(stages):
        name = STAGES[index]
        require(stage["name"] == name and stage["image_id"] == context["image_id"], "Stage order or image differs.")
        command = ["python", "-B", "src/pipeline_stages.py", name]
        if name == "prepare":
            command += ["--raw-dir", context["raw_directory"]]
        command += ["--run-dir", context["run_directory"]]
        if name == "evaluate":
            command += ["--criteria", "/app/src/acceptance_criteria.json"]
        require(stage["command"] == command, "Unexpected stage command.")
        require(type(stage["exit_code"]) is int or
                (stage["exit_code"] is None and isinstance(stage.get("launch_error_type"), str)
                 and bool(stage["launch_error_type"])), "Expected an exit code or an identified launch error.")
        start, end = milliseconds(stage["started_at_utc"]), milliseconds(stage["finished_at_utc"])
        require(previous_end <= start <= end, "Stage times differ from execution order.")
        previous_end = end
        if index < len(stages) - 1:
            require(stage["exit_code"] == 0, "Execution continued after a failed stage.")
    require(len(stages) == 3 or stages[-1]["exit_code"] != 0, "Successful stage prefix is incomplete.")
    return len(stages) == 3 and all(stage["exit_code"] == 0 for stage in stages)


def validate_gate(directory, criteria_path, successful):
    evaluation_path, gate_path = directory / "evaluation_report.json", directory / "acceptance_report.json"
    if not gate_path.exists():
        require(not successful, "Successful workflow has no gate report.")
        return None, {}
    evaluation = read_json(evaluation_path.read_bytes())
    gate = read_json(gate_path.read_bytes())
    require(evaluation["schema_name"] == "reference_run_evaluation_v1"
            and evaluation["evaluation_status"] == "completed"
            and evaluation["run_directory"] == str(directory)
            and evaluation["run_id"] == directory.name
            and evaluation["contract_version"] == CONTRACT_VERSION
            and evaluation["approved_for_delivery"] is False, "Unexpected evaluation identity.")
    criteria, _ = load_criteria(criteria_path)
    model = evaluation_metrics(evaluation, "metrics")
    baseline = evaluation_metrics(evaluation, "training_majority_baseline")
    checks = [
        ("macro_f1", model["macro_f1"], criteria["min_macro_f1"]),
        ("balanced_accuracy", model["balanced_accuracy"], criteria["min_balanced_accuracy"]),
        ("macro_f1_gain_over_training_majority", model["macro_f1"] - baseline["macro_f1"],
         criteria["min_macro_f1_gain_over_training_majority"]),
    ]
    expected = [{"name": name, "observed": value, "minimum": minimum,
                 "passed": value >= minimum - COMPARISON_TOLERANCE} for name, value, minimum in checks]
    passed = all(check["passed"] for check in expected)
    require(gate["schema_name"] == "reference_run_acceptance_v1"
            and gate["scope"] == "team_model_quality_gate"
            and gate["run_id"] == directory.name
            and gate["evaluation_report"] == {"path": str(evaluation_path), "sha256": file_identity(evaluation_path)["sha256"]}
            and gate["criteria"] == {"source": {"path": str(criteria_path), "sha256": file_identity(criteria_path)["sha256"]},
                                    "version": criteria["criteria_version"], "status": criteria["status"]}
            and gate["comparison_tolerance"] == COMPARISON_TOLERANCE
            and gate["checks"] == expected
            and gate["gate_status"] == ("passed" if passed else "failed")
            and gate["model_quality_approval"] == ("TEAM_CRITERIA_MET" if passed else "NOT_ESTABLISHED")
            and gate["approved_for_delivery"] is False, "Gate evidence differs from the agreed criteria.")
    require(not successful or passed, "Successful execution has a failed gate.")
    report = read_json((directory / "validation_report.json").read_bytes())
    require(report["run_directory"] == str(directory) and report["run_id"] == directory.name
            and report["validation_status"] == "passed" and report["approved_for_delivery"] is False
            and report["model_sha256"] == file_identity(directory / "model/trained_model.joblib")["sha256"],
            "Technical report/model identity differs.")
    require(set(evaluation["inputs"]) == {"validation_report.json", "model/trained_model.joblib",
            "preprocessed/X_test.csv", "preprocessed/y_test.csv", "preprocessed/y_train.csv"},
            "Evaluation input selection differs.")
    for relative, identity in evaluation["inputs"].items():
        require(relative in OUTPUTS and identity == {"path": str(directory / relative), **file_identity(directory / relative)},
                "Evaluated input has changed.")
    values = {"evaluation_" + key: value for key, value in model.items()}
    values.update({"training_majority_" + key: value for key, value in baseline.items()})
    values["quality_gate_passed"] = float(passed)
    return gate, values


def record_staged_run(context_path, receipt_path, uri, *, root=None, requirements=None, client=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    context_path, receipt_path = Path(context_path), Path(receipt_path)
    context_bytes, receipt_bytes = context_path.read_bytes(), receipt_path.read_bytes()
    context, receipt = read_json(context_bytes), read_json(receipt_bytes)
    require(context["schema_name"] == "staged_pipeline_invocation_v1"
            and context["evidence_scope"] in ("observed_host_launch", "opaque_test_fixture"), "Unexpected invocation schema.")
    require(re.fullmatch(r"[0-9a-f]{32}", context["invocation_id"]) is not None
            and re.fullmatch(r"[0-9a-f]{40}", context["git_head"]) is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", context["image_id"]) is not None, "Invalid invocation identity.")
    directory = Path(context["run_directory"]).resolve()
    raw = Path(context["raw_directory"]).resolve()
    require(directory.parent == root / "data/runs" and re.fullmatch(r"[A-Za-z0-9_-]+", directory.name) is not None
            and raw == root / "data/raw", "Unexpected input/output location.")
    require(context_path.resolve().parent.name == context["invocation_id"]
            and context_path.resolve().parent.parent == root / "data/run_invocations"
            and receipt_path.resolve() == context_path.resolve().parent / "stage_receipt.json", "Unexpected invocation evidence paths.")
    result_path = directory / "staged_tracking_result.json"
    require(not result_path.exists(), "Tracking result exists; automatic resume is forbidden.")
    require(set(context["source_sha256"]) == set(SOURCES)
            and set(context["configuration_evidence"]) == set(CONFIGURATIONS), "Unexpected evidence selection.")
    files = {}
    for relative in SOURCES:
        path = root / relative
        require(file_identity(path)["sha256"] == context["source_sha256"][relative], "Runtime source differs from launcher observation.")
        files["code/" + relative] = path
    config_bytes = {}
    for relative, value in context["configuration_evidence"].items():
        content = base64.b64decode(value["content_base64"], validate=True)
        require(sha256(content) == value["sha256"], "Configuration snapshot differs.")
        config_bytes[relative] = content
    criteria_path = root / "src/acceptance_criteria.json"
    require(criteria_path.read_bytes() == config_bytes["src/acceptance_criteria.json"], "Runtime criteria differ from launcher observation.")
    require(set(context["raw_identities"]) == set(RAW_FILES)
            and {name: file_identity(raw / name) for name in RAW_FILES} == context["raw_identities"], "Raw inputs changed.")
    successful = validate_receipt(context, receipt)
    gate, metrics = validate_gate(directory, criteria_path, successful)
    for relative in OUTPUTS:
        path = directory / relative
        if successful:
            require(path.is_file(), "Successful workflow output is missing.")
        if path.is_file():
            files["outputs/" + relative] = path
    identities = {name: file_identity(path) for name, path in files.items()}
    requirements_path = Path(requirements or "/opt/requirements-resolved.txt")
    requirements_identity = file_identity(requirements_path)
    evidence_directory = directory / "staged_tracking_evidence"
    require(not evidence_directory.exists(), "Tracking evidence exists; preserve it.")
    evidence_directory.mkdir()
    payloads = {"invocation_context.json": context_bytes, "stage_receipt.json": receipt_bytes,
                "environment/requirements-resolved.txt": requirements_path.read_bytes()}
    require(sha256(payloads["environment/requirements-resolved.txt"]) == requirements_identity["sha256"],
            "Resolved requirements changed during snapshot creation.")
    payloads.update({"configuration/" + key: value for key, value in config_bytes.items()})
    terminal = "FINISHED" if successful else "FAILED"
    state = {"schema_name": "staged_tracking_result_v1", "invocation_id": context["invocation_id"],
             "mlflow_run_id": None, "remote_status": "not_created", "status": "preparing",
             "gate_status": gate["gate_status"] if gate else "not_established", "approved_for_delivery": False}
    client = client or LocalClient(uri)
    known_run = None
    try:
        version = client.server_version()
        require(version == "3.17.0" if client.target == "local" else re.fullmatch(r"3\.\d+\.\d+", version) is not None,
                "Unsupported tracking server version.")
        experiment = "car_accident_staged_pipeline"
        experiments = client.post("experiments/search", {"filter": f"name = '{experiment}'", "view_type": 3, "max_results": 2}).get("experiments", [])
        require(len(experiments) <= 1 and (not experiments or experiments[0]["lifecycle_stage"] == "active"), "Ambiguous or deleted experiment.")
        experiment_id = experiments[0]["experiment_id"] if experiments else client.post("experiments/create", {"name": experiment})["experiment_id"]
        duplicates = client.post("runs/search", {"experiment_ids": [experiment_id], "filter": f"tags.pipeline_invocation_id = '{context['invocation_id']}'", "run_view_type": 3, "max_results": 1}).get("runs", [])
        require(not duplicates, "Invocation already exists; no duplicate or resume.")
        tags = {"pipeline_invocation_id": context["invocation_id"], "pipeline_run_id": directory.name,
                "tracking_scope": "staged_reference_pipeline", "evidence_scope": context["evidence_scope"],
                "launcher_git_head": context["git_head"], "launcher_image_id": context["image_id"],
                "tracking_server_version": version, "tracking_target": client.target,
                "approved_for_delivery": "false", "quality_thresholds_enforced": str(gate is not None).lower(),
                "gate_status": state["gate_status"], "model_quality_approval": gate["model_quality_approval"] if gate else "NOT_ESTABLISHED"}
        state.update(status="unknown", remote_status="unknown")
        created = client.post("runs/create", {"experiment_id": experiment_id, "run_name": directory.name,
                    "start_time": milliseconds(context["observed_at_utc"]), "tags": [{"key": k, "value": v} for k, v in tags.items()]})["run"]
        observed_tags = {v["key"]: v["value"] for v in created["data"].get("tags", [])}
        require(re.fullmatch(r"[0-9a-f]{32}", created["info"]["run_id"]) is not None
                and created["info"]["experiment_id"] == experiment_id and created["info"]["status"] == "RUNNING"
                and all(observed_tags.get(k) == v for k, v in tags.items()), "Unconfirmed run creation; preserve unknown remote state.")
        known_run = created
        state.update(mlflow_run_id=created["info"]["run_id"], remote_status="RUNNING", status="recording")
        provenance = {"schema_name": "staged_pipeline_provenance_v1", "invocation": context,
                      "stage_receipt_sha256": sha256(receipt_bytes), "mlflow_run_id": state["mlflow_run_id"],
                      "artifacts": identities, "terminal_status": terminal, "gate_status": state["gate_status"],
                      "approved_for_delivery": False, "python_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
                      "requirements_identity": requirements_identity,
                      "identity_scope": "Launcher-observed image/commands; artifact recording does not attest the containers independently."}
        payloads["run_provenance.json"] = json_bytes(provenance)
        for name, content in payloads.items():
            path = evidence_directory / name
            write_exclusive(path, content)
            files["evidence/" + name] = path
            identities["evidence/" + name] = file_identity(path)
        for name, path in files.items():
            identity = identities[name]
            require(file_identity(path) == identity, "Artifact changed before upload.")
            client.upload(known_run, name, path, identity["sha256"])
            require(client.download_hash(known_run, name, identity["size_bytes"]) == identity["sha256"], "Artifact round trip differs.")
        require({name: file_identity(raw / name) for name in RAW_FILES} == context["raw_identities"], "Raw inputs changed during recording.")
        require(context_path.read_bytes() == context_bytes and receipt_path.read_bytes() == receipt_bytes, "Invocation evidence changed.")
        require(all(file_identity(path) == identities[name] for name, path in files.items()), "Artifact changed during recording.")
        timestamp = milliseconds(receipt["stages"][-1]["finished_at_utc"])
        if metrics:
            client.post("runs/log-batch", {"run_id": state["mlflow_run_id"], "metrics": [{"key": k, "value": v, "timestamp": timestamp, "step": 0} for k, v in metrics.items()]})
        client.post("runs/update", {"run_id": state["mlflow_run_id"], "status": terminal, "end_time": timestamp})
        observed = client.run(state["mlflow_run_id"])
        actual_tags = {v["key"]: v["value"] for v in observed["data"].get("tags", [])}
        actual_metrics = {v["key"]: v for v in observed["data"].get("metrics", [])}
        require(observed["info"]["run_id"] == state["mlflow_run_id"] and observed["info"]["status"] == terminal
                and observed["info"]["lifecycle_stage"] == "active" and observed["info"]["end_time"] == timestamp
                and all(actual_tags.get(k) == v for k, v in tags.items())
                and all(k in actual_metrics and actual_metrics[k]["value"] == v and actual_metrics[k]["timestamp"] == timestamp
                        and actual_metrics[k]["step"] == 0 for k, v in metrics.items()), "Tracking completion is not confirmed.")
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
                observed = client.run(state["mlflow_run_id"])
                if observed["info"]["run_id"] == state["mlflow_run_id"] and observed["info"]["status"] == "FAILED":
                    state["remote_status"] = "FAILED"
            except Exception as secondary:
                state["secondary_error_type"] = type(secondary).__name__
        if not result_path.exists():
            write_exclusive(result_path, json_bytes(state))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    try:
        result = record_staged_run(args.context, args.receipt, args.tracking_uri)
    except Exception as error:
        detail = ": " + str(error) if isinstance(error, (TrackingFailure, ImportFailure)) else ""
        print("STAGED_TRACKING_FAILED: " + type(error).__name__ + detail)
        return 2
    print(json.dumps(result))
    return 0 if result["remote_status"] == "FINISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
