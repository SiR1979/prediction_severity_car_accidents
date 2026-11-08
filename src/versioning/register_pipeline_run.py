"""Bind one completed staged run to a new private local DVC workspace.

All artifacts remain opaque. This explicit post-registration has no remote
transport, model loading, training, serving selection or delivery approval.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml


DVC_VERSION = "3.67.1"
PREPARED = tuple("preprocessed/" + name + ".csv" for name in
                 ("X_train", "X_test", "y_train", "y_test"))
MODEL = "model/trained_model.joblib"
REPORTS = ("data_validation.json", "validation_report.json", "evaluation_report.json",
           "acceptance_report.json", "metrics.json")
RAW = ("usagers-2021.csv", "caracteristiques-2021.csv", "lieux-2021.csv", "vehicules-2021.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(path):
    """Reject links and Windows reparse points before resolving input paths."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and
                    not (getattr(info, "st_file_attributes", 0) & 0x400),
                    "Linked path rejected: " + str(part))
    return path


def identity(path):
    path = plain(path)
    require(path.is_file(), "Required file missing: " + str(path))
    sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def recorded_identity(path):
    value = identity(path)
    return {key: value[key] for key in ("size_bytes", "sha256")}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=reject_constant)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def pointer_representation(path):
    """Describe bytes; equivalence of pointer text does not identify its data."""
    raw = plain(path).read_bytes()
    return {"raw_sha256": hashlib.sha256(raw).hexdigest(),
            "lf_comparison_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            "lf_comparison_method": "replace_CRLF_bytes_with_LF_only",
            "data_identity_established_by_pointer_text": False}


def utc_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0,
            "Expected UTC evidence time.")
    return result


def validate_inputs(repository, raw, run, provenance_path, receipt_path,
                    invocation_id, run_id, mlflow_run_id):
    """Check original recorded associations against explicit selected local bytes."""
    require(re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is not None and run.name == run_id,
            "Selected run ID differs.")
    require(all(re.fullmatch(r"[0-9a-f]{32}", value) for value in
                (invocation_id, mlflow_run_id)), "Invalid explicit run identity.")
    provenance, receipt = read_json(provenance_path), read_json(receipt_path)
    context = provenance["invocation"]
    require(provenance["schema_name"] == "staged_pipeline_provenance_v1" and
            context["schema_name"] == "staged_pipeline_invocation_v1" and
            receipt["schema_name"] == "staged_pipeline_receipt_v1", "Unexpected evidence schema.")
    require(context["evidence_scope"] in ("observed_host_launch", "opaque_test_fixture"),
            "Unexpected original evidence scope.")
    require(context["invocation_id"] == receipt["invocation_id"] == invocation_id and
            provenance["mlflow_run_id"] == mlflow_run_id and
            PurePosixPath(context["run_directory"]).name == run_id,
            "Original invocation, run or MLflow identity differs.")
    require(provenance["stage_receipt_sha256"] == recorded_identity(receipt_path)["sha256"],
            "Receipt hash differs.")
    require(provenance["terminal_status"] == "FINISHED" and provenance["gate_status"] == "passed"
            and provenance["approved_for_delivery"] is False, "Run is not a completed non-delivery candidate.")
    require(re.fullmatch(r"[0-9a-f]{40}", context["git_head"]) is not None and
            re.fullmatch(r"sha256:[0-9a-f]{64}", context["image_id"]) is not None,
            "Invalid original code/image identity.")
    require(bool(context["source_sha256"]), "Original source identities are missing.")
    for name, digest in context["source_sha256"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                provenance["artifacts"]["code/" + name]["sha256"] == digest,
                "Recorded original source association differs.")
    for value in context["configuration_evidence"].values():
        content = base64.b64decode(value["content_base64"], validate=True)
        require(hashlib.sha256(content).hexdigest() == value["sha256"],
                "Original configuration snapshot hash differs.")
    stages = receipt["stages"]
    require(isinstance(stages, list) and len(stages) == 3, "Completed run needs three stages.")
    previous = utc_time(context["observed_at_utc"])
    for name, stage in zip(("prepare", "train", "evaluate"), stages):
        command = ["python", "-B", "src/pipeline_stages.py", name]
        if name == "prepare":
            command += ["--raw-dir", context["raw_directory"]]
        command += ["--run-dir", context["run_directory"]]
        if name == "evaluate":
            command += ["--criteria", "/app/src/acceptance_criteria.json"]
        start, end = utc_time(stage["started_at_utc"]), utc_time(stage["finished_at_utc"])
        require(stage["name"] == name and stage["image_id"] == context["image_id"] and
                type(stage["exit_code"]) is int and stage["exit_code"] == 0 and
                stage.get("launch_error_type") is None and stage["command"] == command and
                previous <= start <= end, "Stage completion or original execution identity differs.")
        previous = end
    require(set(context["raw_identities"]) == set(RAW), "Raw input selection differs.")
    files = {"provenance": provenance_path, "receipt": receipt_path}
    for name in RAW:
        require(recorded_identity(raw / name) == context["raw_identities"][name],
                "Actual raw input differs: " + name)
        files["raw/" + name] = raw / name
    for name in (*PREPARED, MODEL, *REPORTS):
        require(recorded_identity(run / name) == provenance["artifacts"]["outputs/" + name],
                "Actual run artifact differs: " + name)
        files["outputs/" + name] = run / name
    validation = read_json(run / "validation_report.json")
    evaluation = read_json(run / "evaluation_report.json")
    gate = read_json(run / "acceptance_report.json")
    for report in (validation, evaluation):
        require(report["run_id"] == run_id and report["run_directory"] == context["run_directory"]
                and report["approved_for_delivery"] is False, "Report belongs to another run.")
    require(validation["validation_status"] == "passed" and
            validation["model_sha256"] == recorded_identity(run / MODEL)["sha256"] and
            validation["model_validation"]["model_sha256"] == validation["model_sha256"] and
            evaluation["evaluation_status"] == "completed", "Technical/evaluation report differs.")
    expected_inputs = {"validation_report.json", MODEL, "preprocessed/X_test.csv",
                       "preprocessed/y_test.csv", "preprocessed/y_train.csv"}
    require(set(evaluation["inputs"]) == expected_inputs, "Evaluation input selection differs.")
    for name in expected_inputs:
        require(evaluation["inputs"][name] == {
            "path": context["run_directory"] + "/" + name, **recorded_identity(run / name)},
            "Evaluated artifact association differs: " + name)
    require(gate["run_id"] == run_id and gate["gate_status"] == "passed" and
            gate["approved_for_delivery"] is False and gate["checks"] and
            all(check["passed"] is True for check in gate["checks"]) and
            gate["evaluation_report"] == {
                "path": context["run_directory"] + "/evaluation_report.json",
                "sha256": recorded_identity(run / "evaluation_report.json")["sha256"]},
            "Recorded gate association differs.")
    criteria = context["configuration_evidence"]["src/acceptance_criteria.json"]
    require(gate["criteria"]["source"]["sha256"] == criteria["sha256"],
            "Recorded gate criteria association differs.")
    configuration = validation["model_validation"].get("estimator_configuration")
    parameters = {"status": "missing_in_original_validation_report"}
    if configuration is not None:
        require(configuration["method"] == "get_params(deep=False)" and
                isinstance(configuration["parameters"], dict), "Invalid recorded parameter evidence.")
        parameters = {"status": "recorded_in_original_validation_report", "evidence": configuration}
    return provenance, parameters, files


class LocalDvc:
    def __init__(self, workspace):
        self.workspace, self.sequence = workspace, 0
        # Only runtime necessities cross the process boundary; no DVC/Git/auth overrides.
        allowed = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
                   "LANG", "LC_ALL", "VIRTUAL_ENV", "PYTHONHOME", "USERNAME")
        self.env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        for key, relative in {
            "DVC_GLOBAL_CONFIG_DIR": "isolation/global", "DVC_SYSTEM_CONFIG_DIR": "isolation/system",
            "DVC_SITE_CACHE_DIR": "isolation/site", "HOME": "isolation/home",
            "USERPROFILE": "isolation/home", "APPDATA": "isolation/appdata",
            "LOCALAPPDATA": "isolation/localappdata", "TMP": "isolation/tmp",
            "TEMP": "isolation/tmp", "TMPDIR": "isolation/tmp",
            "XDG_CACHE_HOME": "isolation/xdg_cache", "XDG_CONFIG_HOME": "isolation/xdg_config",
        }.items():
            directory = workspace / relative
            directory.mkdir(parents=True, exist_ok=True)
            self.env[key] = str(directory)
        self.env.update(DVC_NO_ANALYTICS="true", DVC_STUDIO_OFFLINE="true",
                        DVC_EXP_AUTO_PUSH="false", PYTHONDONTWRITEBYTECODE="1")

    def command(self, *args):
        self.sequence += 1
        argv = [sys.executable, "-B", "-m", "dvc", *args]
        record = {"argv": argv, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            process = subprocess.run(argv, cwd=self.workspace, env=self.env,
                                     capture_output=True, text=True, timeout=120)
            record.update(exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(self.workspace / ("command_%02d.json" % self.sequence), record)
        require(process.returncode == 0, "Local DVC command failed; exit=%s; argv=%s; %s" % (
            process.returncode, argv, (process.stderr or process.stdout)[-6000:]))
        return process.stdout.strip()


def register_pipeline_run(*, repository, raw_directory, run_directory, provenance_path,
                          receipt_path, workspace, invocation_id, run_id, mlflow_run_id,
                          original_pointers=()):
    original_pointers = tuple(original_pointers)
    repository, raw, run, provenance_path, receipt_path, workspace = map(plain, (
        repository, raw_directory, run_directory, provenance_path, receipt_path, workspace))
    require(repository.is_dir() and run.is_dir() and raw.is_dir(), "Selected input directories are missing.")
    require(not workspace.exists(), "Existing workspace preserved; no resume or retry.")
    require(workspace.parent.is_dir(), "Workspace parent must already exist.")
    for protected in (repository, raw, run, provenance_path, receipt_path,
                      *(plain(path) for path in original_pointers)):
        require(workspace != protected and workspace not in protected.parents and
                protected not in workspace.parents, "Workspace overlaps a protected input.")
    require(importlib.metadata.version("dvc") == DVC_VERSION, "Expected existing DVC " + DVC_VERSION)
    provenance, parameters, files = validate_inputs(
        repository, raw, run, provenance_path, receipt_path, invocation_id, run_id, mlflow_run_id)
    pointer_inputs = []
    for index, path in enumerate(original_pointers):
        path = plain(path)
        require(path.suffix == ".dvc", "Original pointer must have .dvc suffix.")
        files["original_pointer/%d" % index] = path
        pointer_inputs.append({"path": str(path), **pointer_representation(path),
                               "underlying_data_association": "not_established_by_this_pointer_comparison"})
    before = {name: identity(path) for name, path in files.items()}
    source_path = plain(__file__)
    registration_source = identity(source_path)
    # All checks above are read-only. Atomically claim a fresh private workspace.
    workspace.mkdir(mode=0o700)
    started = datetime.now(timezone.utc).isoformat()
    try:
        dvc = LocalDvc(workspace)
        require(dvc.command("--version") == DVC_VERSION, "DVC command version differs.")
        dvc.command("init", "--no-scm")
        dvc.command("config", "core.analytics", "false")
        dvc.command("config", "cache.type", "copy")
        dvc.command("config", "cache.dir", str(workspace / ".dvc/cache"))
        require(not dvc.command("remote", "list"), "Isolated workspace inherited a remote.")
        evidence = workspace / "evidence"
        evidence.mkdir()
        for name, path in (("run_provenance.json", provenance_path), ("stage_receipt.json", receipt_path)):
            with (evidence / name).open("xb") as destination, path.open("rb") as source:
                shutil.copyfileobj(source, destination)
            require(identity(evidence / name) == before["provenance" if name.startswith("run_") else "receipt"],
                    "Evidence copy differs.")
        objects = {}
        for name in (*PREPARED, MODEL):
            target = workspace / "artifacts" / run_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as destination, (run / name).open("rb") as source:
                shutil.copyfileobj(source, destination)
            require(identity(target) == before["outputs/" + name], "Opaque artifact copy differs.")
            relative = target.relative_to(workspace).as_posix()
            dvc.command("add", "--no-relink", relative)
            pointer = target.with_name(target.name + ".dvc")
            document = yaml.safe_load(pointer.read_bytes())
            require(set(document) == {"outs"} and len(document["outs"]) == 1,
                    "Unexpected generated DVC pointer scope.")
            output = document["outs"][0]
            observed = identity(target)
            require(output["path"] == target.name and output["md5"] == observed["md5"] and
                    output["size"] == observed["size_bytes"] and output.get("hash", "md5") == "md5",
                    "DVC pointer differs from copied artifact bytes.")
            md5 = output["md5"]
            cache = workspace / ".dvc/cache/files/md5" / md5[:2] / md5[2:]
            require(identity(cache) == observed, "Isolated DVC cache bytes differ.")
            objects[name] = {**observed, "workspace_path": relative,
                             "pointer_path": pointer.relative_to(workspace).as_posix(),
                             "pointer_representation": pointer_representation(pointer),
                             "dvc_object": {"algorithm": "md5", "value": md5,
                                            "local_cache_path": cache.relative_to(workspace).as_posix()}}
        require(all(identity(path) == before[name] for name, path in files.items()),
                "Protected input changed during registration.")
        require(identity(source_path) == registration_source, "Registrar source changed.")
        manifest = {
            "schema_name": "staged_run_local_dvc_link_v1", "status": "registered_locally",
            "run_id": run_id, "invocation_id": invocation_id, "mlflow_run_id": mlflow_run_id,
            "original_training": provenance["invocation"],
            "original_recorded_artifacts": provenance["artifacts"],
            "original_evidence_identities": {key: before[key] for key in ("receipt", "provenance")},
            "selected_input_identities": before, "original_pointer_representations": pointer_inputs,
            "estimator_configuration_evidence": parameters, "objects": objects,
            "registration": {"started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                             "source_path": str(source_path), "source_identity": registration_source,
                             "python": sys.version, "dvc_version": DVC_VERSION,
                             "workspace": str(workspace), "scm": "none", "cache_type": "copy"},
            "recorded_terminal_status": provenance["terminal_status"],
            "recorded_gate_status": provenance["gate_status"],
            "remote_confirmation": "not_attempted", "remote_transfer": "not_attempted",
            "training_performed": False, "model_deserialized": False,
            "approved_for_delivery": False,
        }
        write_json(workspace / "link_manifest.json", manifest)
        return manifest
    except Exception as error:
        write_json(workspace / "registration_failure.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "started_at_utc": started, "approved_for_delivery": False,
            "recovery": "Preserve this workspace and diagnostics; no automatic retry."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "raw-directory", "run-directory", "provenance-path", "receipt-path", "workspace"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("invocation-id", "run-id", "mlflow-run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--original-pointer", type=Path, action="append", default=[])
    args = vars(parser.parse_args())
    args["original_pointers"] = args.pop("original_pointer")
    try:
        result = register_pipeline_run(**args)
    except Exception as error:
        print("LOCAL_REGISTRATION_FAILED: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "workspace": result["registration"]["workspace"],
                      "approved_for_delivery": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
