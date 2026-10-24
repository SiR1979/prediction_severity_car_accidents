"""Record new reference workflows through the local or DagsHub artifact proxy.

Tracking transfers model bytes without deserialisation. Historical reference
import contracts are not used by this module. New-run evaluation is
linked to the same run and verified before workflow completion.
"""

import base64
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

from .import_reference_run import LocalClient


EXPERIMENT = "car_accident_reference_pipeline"
SOURCE_FILES = (
    "src/run_pipeline.py", "src/evaluate_run.py", "src/reference_contract.py",
    "src/data/make_dataset.py", "src/models/train_model.py",
    "src/tracking/track_pipeline_run.py", "src/tracking/import_reference_run.py",
)
CONFIG_FILES = (
    "Dockerfile", "requirements-runtime.txt", "compose.yaml", "compose.tracking.yaml",
    "scripts/run_tracked_pipeline.ps1",
)
RAW_FILES = ("usagers-2021.csv", "caracteristiques-2021.csv", "lieux-2021.csv", "vehicules-2021.csv")
LOCAL_MLFLOW_VERSION = "3.17.0"


class TrackingFailure(RuntimeError):
    """A new run cannot be confirmed in the local tracking store."""


def require(condition, message):
    if not condition:
        raise TrackingFailure(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "Missing or linked evidence file.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def read_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise TrackingFailure("Non-finite JSON constant.")

    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)


def write_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


class PipelineTracker:
    def __init__(self, uri, context_path, *, storage_root=None, versioning_directory=None,
                 resolved_requirements=None, experiment=EXPERIMENT):
        self.client = LocalClient(uri)
        self.target = self.client.target
        self.server_version = None
        self.context_path = Path(context_path)
        self.source_root = Path(__file__).resolve().parents[2]
        self.storage_root = Path(storage_root or self.source_root).resolve()
        self.versioning_directory = Path(versioning_directory or self.storage_root / "versioning")
        self.resolved_requirements = Path(resolved_requirements or "/opt/requirements-resolved.txt")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", experiment) is not None, "Invalid experiment name.")
        self.experiment = experiment
        self.run = None
        self.state = {"status": "not_started", "phase": "tracking_start", "mlflow_run_id": None,
                      "invocation_id": None, "remote_status": "not_created"}

    def check_version(self):
        version = self.client.server_version()
        if self.target == "local":
            require(version == LOCAL_MLFLOW_VERSION, "Expected local MLflow 3.17.0 server.")
        else:
            # DagsHub upgrades its server independently; accept MLflow 3.x and record the observed version.
            require(re.fullmatch(r"3\.\d+\.\d+", version) is not None, "Expected an MLflow 3.x server on DagsHub.")
        self.server_version = version

    def capture_context(self, raw_directory, run_directory):
        require(raw_directory == self.storage_root / "data/raw", "Tracked raw directory is outside the reference input location.")
        require(run_directory.parent == self.storage_root / "data/runs", "Tracked run directory is outside data/runs.")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", run_directory.name) is not None, "Unsafe run name.")
        identity = file_identity(self.context_path)
        require(identity["size_bytes"] <= 1024 * 1024, "Invocation context is too large.")
        self.context_bytes = self.context_path.read_bytes()
        require(sha256(self.context_bytes) == identity["sha256"], "Context changed while reading.")
        context = read_json(self.context_bytes)
        require(context["schema_name"] == "reference_pipeline_invocation_v1", "Unknown invocation schema.")
        require(context["evidence_scope"] in ("observed_host_launch", "opaque_test_fixture"), "Unknown evidence scope.")
        require(re.fullmatch(r"[0-9a-f]{32}", context["invocation_id"]) is not None, "Invalid invocation identity.")
        self.state["invocation_id"] = context["invocation_id"]
        require(re.fullmatch(r"[0-9a-f]{40}", context["git_head"]) is not None, "Invalid observed Git baseline.")
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", context["image_id"]) is not None, "Invalid observed image identity.")
        require(isinstance(context["worktree_status"], list) and all(isinstance(v, str) for v in context["worktree_status"]), "Invalid worktree observation.")
        require(datetime.fromisoformat(context["observed_at_utc"]).utcoffset().total_seconds() == 0, "Invocation timestamp is not UTC.")
        require(context["raw_directory"] == str(raw_directory) and context["run_directory"] == str(run_directory), "Invocation paths differ.")
        require(set(context["source_sha256"]) == set(SOURCE_FILES), "Unexpected source selection.")
        self.source_bytes = {}
        for relative in SOURCE_FILES:
            path = self.source_root / relative
            file_identity(path)
            content = path.read_bytes()
            require(sha256(content) == context["source_sha256"][relative], f"Executed source differs from launcher observation: {relative}")
            self.source_bytes[relative] = content
        require(set(context["configuration_evidence"]) == set(CONFIG_FILES), "Unexpected configuration selection.")
        self.config_bytes = {}
        for relative, evidence in context["configuration_evidence"].items():
            content = base64.b64decode(evidence["content_base64"], validate=True)
            require(sha256(content) == evidence["sha256"], "Configuration evidence hash differs.")
            self.config_bytes[relative] = content
        self.raw_directory = raw_directory
        self.raw_identities = {name: file_identity(raw_directory / name) for name in RAW_FILES}
        requirements_identity = file_identity(self.resolved_requirements)
        self.requirements_bytes = self.resolved_requirements.read_bytes()
        require(sha256(self.requirements_bytes) == requirements_identity["sha256"], "Resolved dependency evidence changed.")
        self.context = context
        self.environment = {
            "python_version": platform.python_version(), "platform": platform.platform(),
            "installed_packages": sorted(f"{distribution.metadata['Name']}=={distribution.version}" for distribution in importlib.metadata.distributions()),
            "resolved_requirements": self.requirements_bytes.decode("utf-8").splitlines(),
            "resolved_requirements_sha256": requirements_identity["sha256"],
            "launcher_observed_image_id": context["image_id"],
        }

    def dvc_reference(self):
        pointer = self.versioning_directory / "raw_2021.dvc"
        manifest = self.versioning_directory / "reference_manifest.json"
        if not pointer.is_file() or not manifest.is_file():
            return {"association_status": "unavailable", "reason": "Reference metadata unavailable", "object_id": None}
        pointer_identity, manifest_identity = file_identity(pointer), file_identity(manifest)
        pointer_text = pointer.read_text(encoding="utf-8")
        metadata = read_json(manifest.read_bytes())
        object_match = re.search(r"(?m)^\s+md5: ([0-9a-f]{32}\.dir)\s*$", pointer_text)
        require(object_match is not None and re.search(r"(?m)^- path: data/raw\s*$", pointer_text) is not None, "Unexpected raw DVC pointer.")
        matched = pointer_identity["sha256"] == metadata["metadata_file_sha256"].get("versioning/raw_2021.dvc")
        matched = matched and all(metadata["source_file_sha256"].get("data/raw/" + name) == identity for name, identity in self.raw_identities.items())
        return {"association_status": "matched" if matched else "unmatched", "object_id": object_match.group(1),
                "pointer": "versioning/raw_2021.dvc", "pointer_sha256": pointer_identity["sha256"],
                "manifest_sha256": manifest_identity["sha256"], "declared_dvc_version": metadata["dvc_version"],
                "registration_baseline_git_head": metadata["registration_baseline_git_head"],
                "scope": "existing_input_reference_only; no new output registration or recovery verification"}

    def start(self, raw_directory, run_directory, record, features, contract_version):
        self.directory = run_directory
        self.started_at = record["started_at_utc"]
        self.capture_context(raw_directory, run_directory)
        self.dvc = self.dvc_reference()
        self.configuration = {"split_random_state": 42, "forest_random_state": None, "forest_n_jobs": -1,
                              "target_mapping": [1, 1, 1, 0], "original_target_codes": [1, 2, 3, 4],
                              "features": list(features), "contract_version": contract_version}
        self.check_version()
        experiments = self.client.post("experiments/search", {"filter": f"name = '{self.experiment}'", "view_type": 3, "max_results": 2}).get("experiments", [])
        require(len(experiments) <= 1, "Ambiguous experiment; stored state preserved.")
        if experiments:
            require(experiments[0]["lifecycle_stage"] == "active", "Deleted experiment preserved.")
            experiment_id = experiments[0]["experiment_id"]
        else:
            experiment_id = self.client.post("experiments/create", {"name": self.experiment})["experiment_id"]
        duplicates = self.client.post("runs/search", {"experiment_ids": [experiment_id], "filter": f"tags.pipeline_invocation_id = '{self.state['invocation_id']}'", "run_view_type": 3, "max_results": 1}).get("runs", [])
        require(not duplicates, "Invocation already recorded; no resume or duplicate creation.")
        self.tags = {"tracking_scope": "new_reference_pipeline_run", "pipeline_invocation_id": self.state["invocation_id"],
                     "evidence_scope": self.context["evidence_scope"],
                     "pipeline_run_id": run_directory.name, "pipeline_phase": "tracking_start",
                     "launcher_git_head": self.context["git_head"], "launcher_image_id": self.context["image_id"],
                     "invocation_context_sha256": sha256(self.context_bytes), "approved_for_delivery": "false",
                     "quality_thresholds_enforced": "false", "evaluation_performed": "false",
                     "model_quality_approval": "NOT_ESTABLISHED",
                     "tracking_target": self.target}
        if self.server_version is not None:
            self.tags["tracking_server_version"] = self.server_version
        self.state.update(status="unknown", remote_status="unknown")
        created = self.client.post("runs/create", {"experiment_id": experiment_id, "run_name": run_directory.name,
                                  "start_time": int(datetime.fromisoformat(self.started_at).timestamp() * 1000),
                                  "tags": [{"key": key, "value": value} for key, value in self.tags.items()]})
        new_run = created["run"]
        run_id = new_run["info"]["run_id"]
        require(re.fullmatch(r"[0-9a-f]{32}", run_id) is not None, "Invalid returned run identity; creation outcome is uncertain.")
        returned_tags = {item["key"]: item["value"] for item in new_run["data"].get("tags", [])}
        require(all(returned_tags.get(key) == value for key, value in self.tags.items())
                and new_run["info"]["status"] == "RUNNING"
                and new_run["info"]["experiment_id"] == experiment_id
                and new_run["info"]["start_time"] == int(datetime.fromisoformat(self.started_at).timestamp() * 1000),
                "Create response is not the new invocation; preserve uncertain remote state.")
        LocalClient.artifact_route(new_run, "pipeline/run_provenance.json")
        self.run = new_run
        self.state.update(mlflow_run_id=run_id, status="running", remote_status="RUNNING")
        self.params = {"split_random_state": "42", "forest_random_state": "None", "forest_n_jobs": "-1",
                  "feature_count": str(len(features)), "contract_version": contract_version,
                  "target_mapping": "[1,2,3,4] -> [1,1,1,0]"}
        self.client.post("runs/log-batch", {"run_id": run_id, "params": [{"key": key, "value": value} for key, value in self.params.items()]})

    def phase(self, name):
        self.state["phase"] = name
        if self.state["mlflow_run_id"] is not None:
            self.set_tags({"pipeline_phase": name, "phase_" + name + "_at_utc": utc_now()})

    def set_tags(self, values):
        for key, value in values.items():
            self.client.post("runs/set-tag", {"run_id": self.state["mlflow_run_id"], "key": key, "value": str(value)})
            self.tags[key] = str(value)

    def check_inputs_unchanged(self):
        for relative, content in self.source_bytes.items():
            require(file_identity(self.source_root / relative)["sha256"] == sha256(content), "Source changed during execution.")
        require({name: file_identity(self.raw_directory / name) for name in RAW_FILES} == self.raw_identities, "Raw input changed during execution.")
        require(file_identity(self.context_path)["sha256"] == sha256(self.context_bytes), "Invocation context changed during execution.")

    def finish(self, report_path, model_path, evaluation_path):
        self.check_inputs_unchanged()
        require(Path(model_path).resolve() == self.directory / "model/trained_model.joblib", "Model is outside the new run.")
        require(Path(report_path).resolve() == self.directory / "validation_report.json", "Technical report is outside the new run.")
        require(Path(evaluation_path).resolve() == self.directory / "evaluation_report.json", "Evaluation report is outside the new run.")
        files = {"pipeline/validation_report.json": Path(report_path), "pipeline/model/trained_model.joblib": Path(model_path),
                 "pipeline/evaluation_report.json": Path(evaluation_path)}
        for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
            files["pipeline/preprocessed/" + name] = self.directory / "preprocessed" / name
        # Capture one baseline; subsequent observations only check for changes.
        identities = {key: file_identity(path) for key, path in files.items()}
        report_bytes = Path(report_path).read_bytes()
        report_identity = identities["pipeline/validation_report.json"]
        require({"sha256": sha256(report_bytes), "size_bytes": len(report_bytes)} == report_identity, "Technical report changed while reading.")
        report = read_json(report_bytes)
        require(report["validation_status"] == "passed" and report["approved_for_delivery"] is False, "Technical report is not accepted.")
        require(report["run_directory"] == str(self.directory) and report["run_id"] == self.directory.name, "Technical report belongs to another run.")
        model_identity = identities["pipeline/model/trained_model.joblib"]
        require(report["model_sha256"] == model_identity["sha256"] and report["model_path"] == str(model_path), "Technical model identity differs.")
        require(report["model_validation"]["model_sha256"] == model_identity["sha256"]
                and report["model_validation"]["model_path"] == str(model_path), "Validated model identity differs.")
        evaluation_bytes = Path(evaluation_path).read_bytes()
        evaluation_identity = identities["pipeline/evaluation_report.json"]
        require({"sha256": sha256(evaluation_bytes), "size_bytes": len(evaluation_bytes)} == evaluation_identity,
                "Evaluation report changed while reading.")
        evaluation = read_json(evaluation_bytes)
        require(evaluation["schema_name"] == "reference_run_evaluation_v1"
                and evaluation["scope"] == "descriptive_prepared_holdout_evaluation"
                and evaluation["evaluation_status"] == "completed", "Evaluation is not complete.")
        require(evaluation["run_id"] == self.directory.name and evaluation["run_directory"] == str(self.directory)
                and evaluation["contract_version"] == self.configuration["contract_version"], "Evaluation belongs to another run or contract.")
        require(evaluation["approved_for_delivery"] is False and evaluation["quality_thresholds_enforced"] is False
                and evaluation["model_quality_approval"] == "NOT_ESTABLISHED", "Evaluation approval boundaries differ.")
        expected_inputs = {"validation_report.json", "model/trained_model.joblib", "preprocessed/X_test.csv",
                           "preprocessed/y_test.csv", "preprocessed/y_train.csv"}
        require(set(evaluation["inputs"]) == expected_inputs, "Unexpected evaluated input selection.")
        for relative, observed in evaluation["inputs"].items():
            expected = {"path": str(self.directory / relative), **identities["pipeline/" + relative]}
            require(observed == expected, "Evaluated input identity differs from the recorded artifact.")
        require(evaluation["test_rows"] == report["data_validation"]["partitions"]["test"]["row_count"]
                and evaluation["feature_count"] == len(self.configuration["features"]), "Evaluation dimensions differ.")
        baseline = evaluation["training_majority_baseline"]
        counts = baseline["training_class_counts"]
        require(set(counts) == {"0", "1"} and all(type(value) is int and value >= 0 for value in counts.values())
                and sum(counts.values()) == report["data_validation"]["partitions"]["train"]["row_count"],
                "Training-majority counts differ.")
        require(type(baseline["class_code"]) is int and baseline["class_code"] in (0, 1)
                and baseline["selection_basis"] == "training_targets_only" and baseline["tie_break"] == "lowest_class_code"
                and baseline["class_code"] == (0 if counts["0"] >= counts["1"] else 1), "Training-majority selection differs.")
        metric_names = {"accuracy", "balanced_accuracy", "macro_f1"}
        self.metrics = {}
        for prefix, values in (("evaluation", evaluation["metrics"]), ("training_majority", baseline["metrics"])):
            require(set(values) == metric_names, "Unexpected evaluation metric selection.")
            for name, value in values.items():
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "Invalid evaluation metric.")
                self.metrics[prefix + "_" + name] = value
        evaluation_link = {"artifact": "pipeline/evaluation_report.json", **evaluation_identity,
                           "metrics": evaluation["metrics"], "training_majority_baseline": baseline,
                           "inputs": evaluation["inputs"]}
        provenance = {"schema_name": "reference_pipeline_provenance_v1", "scope": "new_reference_pipeline_run",
                      "evidence_scope": self.context["evidence_scope"],
                      "run_id": self.directory.name, "run_directory": str(self.directory),
                      "invocation_id": self.state["invocation_id"], "mlflow_run_id": self.state["mlflow_run_id"],
                      "started_at_utc": self.started_at, "captured_at_utc": utc_now(),
                      "launcher_git_head": self.context["git_head"], "launcher_worktree_status": self.context["worktree_status"],
                      "executed_source_sha256": {key: sha256(value) for key, value in self.source_bytes.items()},
                      "configuration_evidence_sha256": {key: sha256(value) for key, value in self.config_bytes.items()},
                      "raw_file_identities": self.raw_identities, "configuration": self.configuration,
                      "environment": self.environment, "dvc_input_reference": self.dvc,
                      "model_sha256": model_identity["sha256"], "technical_report_sha256": report_identity["sha256"],
                      "approved_for_delivery": False, "model_quality_approval": "NOT_ESTABLISHED",
                      "evaluation_performed": True, "evaluation": evaluation_link,
                      "git_identity_scope": "Launcher-observed baseline; executed bytes are identified independently.",
                      "image_identity_scope": "Launcher-observed immutable image ID; not an in-container attestation."}
        evidence = {"run_provenance.json": json_bytes(provenance), "invocation_context.json": self.context_bytes,
                    "environment/requirements-resolved.txt": self.requirements_bytes}
        evidence.update({"code/" + key: value for key, value in self.source_bytes.items()})
        evidence.update({"configuration/" + key: value for key, value in self.config_bytes.items()})
        for relative, content in evidence.items():
            path = self.directory / "tracking_evidence" / relative
            write_exclusive(path, content)
            files["pipeline/" + relative] = path
            identities["pipeline/" + relative] = {"sha256": sha256(content), "size_bytes": len(content)}
        manifest = {"schema_name": "reference_pipeline_artifacts_v1", "run_id": self.directory.name,
                    "mlflow_run_id": self.state["mlflow_run_id"], "invocation_id": self.state["invocation_id"],
                    "created_at_utc": utc_now(), "model_sha256": model_identity["sha256"],
                    "approved_for_delivery": False, "model_quality_approval": "NOT_ESTABLISHED",
                    "evaluation_report_sha256": evaluation_identity["sha256"], "artifacts": dict(identities)}
        manifest_path = self.directory / "tracking_evidence/artifact_manifest.json"
        manifest_bytes = json_bytes(manifest)
        write_exclusive(manifest_path, manifest_bytes)
        files["pipeline/artifact_manifest.json"] = manifest_path
        identities["pipeline/artifact_manifest.json"] = {"sha256": sha256(manifest_bytes), "size_bytes": len(manifest_bytes)}
        self.set_tags({"model_sha256": model_identity["sha256"], "technical_report_sha256": provenance["technical_report_sha256"],
                       "provenance_sha256": identities["pipeline/run_provenance.json"]["sha256"],
                       "artifact_manifest_sha256": identities["pipeline/artifact_manifest.json"]["sha256"],
                       "evaluation_performed": "true", "evaluation_report_sha256": evaluation_identity["sha256"]})
        for name, path in files.items():
            identity = identities[name]
            require(file_identity(path) == identity, "Local artifact changed before recording.")
            self.client.upload(self.run, name, path, identity["sha256"])
            require(self.client.download_hash(self.run, name, identity["size_bytes"]) == identity["sha256"], "Stored artifact SHA-256 differs.")
            require(file_identity(path) == identity, "Local artifact changed during recording.")
        self.check_inputs_unchanged()
        evaluated_at = datetime.fromisoformat(evaluation["checked_at_utc"])
        require(evaluated_at.utcoffset() is not None and evaluated_at.utcoffset().total_seconds() == 0,
                "Evaluation timestamp is not UTC.")
        self.metric_timestamp = int(evaluated_at.timestamp() * 1000)
        self.client.post("runs/log-batch", {"run_id": self.state["mlflow_run_id"], "metrics": [
            {"key": key, "value": value, "timestamp": self.metric_timestamp, "step": 0}
            for key, value in self.metrics.items()
        ]})
        self.phase("tracking_finish")
        ended_at = utc_now()
        end_time = int(datetime.fromisoformat(ended_at).timestamp() * 1000)
        self.client.post("runs/update", {"run_id": self.state["mlflow_run_id"], "status": "FINISHED", "end_time": end_time})
        self.state["remote_status"] = "unknown"
        observed = self.client.run(self.state["mlflow_run_id"])
        require(observed["info"]["run_id"] == self.state["mlflow_run_id"] and observed["info"]["status"] == "FINISHED"
                and observed["info"]["lifecycle_stage"] == "active" and observed["info"]["end_time"] == end_time, "Remote completion is not confirmed.")
        stored_tags = {item["key"]: item["value"] for item in observed["data"].get("tags", [])}
        require(all(stored_tags.get(key) == value for key, value in self.tags.items()), "Recorded metadata differs.")
        stored_params = {item["key"]: item["value"] for item in observed["data"].get("params", [])}
        require(stored_params == self.params, "Recorded configuration parameters differ.")
        stored_metrics = {item["key"]: item for item in observed["data"].get("metrics", [])}
        require(all(key in stored_metrics and stored_metrics[key]["value"] == value
                    and stored_metrics[key].get("timestamp") == self.metric_timestamp
                    and stored_metrics[key].get("step") == 0 for key, value in self.metrics.items()),
                "Recorded evaluation metrics differ.")
        for name, path in files.items():
            require(file_identity(path) == identities[name], "Local artifact changed before completion.")
        self.state.update(status="recorded", remote_status="FINISHED", verified_at_utc=utc_now(),
                          artifact_manifest_sha256=identities["pipeline/artifact_manifest.json"]["sha256"],
                          evaluation_report_sha256=evaluation_identity["sha256"])

    def fail(self, error):
        self.state["failed_phase"] = self.state["phase"]
        self.state["status"] = "failed"
        if self.state["mlflow_run_id"] is None:
            return
        secondary = []
        try:
            self.set_tags({"pipeline_failed_phase": self.state["phase"], "pipeline_error_type": type(error).__name__,
                           "pipeline_failed_at_utc": utc_now()})
        except Exception as tracking_error:
            secondary.append(type(tracking_error).__name__)
        try:
            self.client.post("runs/update", {"run_id": self.state["mlflow_run_id"], "status": "FAILED",
                                            "end_time": int(datetime.now(timezone.utc).timestamp() * 1000)})
            observed = self.client.run(self.state["mlflow_run_id"])
            require(observed["info"]["run_id"] == self.state["mlflow_run_id"] and observed["info"]["status"] == "FAILED", "Remote failure is not confirmed.")
            self.state["remote_status"] = "FAILED"
        except Exception as tracking_error:
            self.state["remote_status"] = "unknown"
            secondary.append(type(tracking_error).__name__)
        if secondary:
            self.state["secondary_error_types"] = secondary
