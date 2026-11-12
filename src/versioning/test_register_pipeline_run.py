"""Offline identity and local DVC fixtures; no fitted or deserialized models."""

import base64
import copy
import getpass
import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import run_pipeline as pipeline
from versioning import register_pipeline_run as registrar


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        # The caller supplies a fresh evidence root outside the repository.
        parent = os.environ.get("RUN_VERSION_TEST_ROOT")
        self.assertTrue(parent, "RUN_VERSION_TEST_ROOT must select a private fixture root.")
        self.root = Path(tempfile.mkdtemp(prefix="fixture_", dir=parent))
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.raw = self.repo / "data/raw"
        self.run_id, self.invocation, self.mlflow = "staged_fixture", "1" * 32, "2" * 32
        self.run = self.repo / "data/runs" / self.run_id
        self.run.mkdir(parents=True)
        self.raw.mkdir(parents=True)
        self.workspace = self.root / "private_registration"
        self.provenance_path, self.receipt_path = self.root / "provenance.json", self.root / "receipt.json"
        self.pointer = self.repo / "historical.dvc"
        self.pointer.write_bytes(b"outs:\r\n- md5: 00000000000000000000000000000000\r\n  path: old.csv\r\n")
        for name in registrar.RAW:
            (self.raw / name).write_bytes(b"opaque raw fixture\x00" + name.encode())
        for name in (*registrar.PREPARED, registrar.MODEL):
            target = self.run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"opaque artifact fixture\x00\r\n" + name.encode())
        self.recorded_run = "/app/data/runs/" + self.run_id
        self.configuration = {"criteria_version": "fixture", "status": "test_only"}
        config_bytes = json.dumps(self.configuration).encode()
        self.context = {
            "schema_name": "staged_pipeline_invocation_v1", "evidence_scope": "opaque_test_fixture",
            "invocation_id": self.invocation, "git_head": "3" * 40, "image_id": "sha256:" + "4" * 64,
            "observed_at_utc": "2026-11-07T10:00:00+00:00", "raw_directory": "/app/data/raw",
            "run_directory": self.recorded_run, "source_sha256": {"src/run_pipeline.py": "5" * 64},
            "raw_identities": {name: registrar.recorded_identity(self.raw / name) for name in registrar.RAW},
            "configuration_evidence": {"src/acceptance_criteria.json": {
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
                "content_base64": base64.b64encode(config_bytes).decode()}},
        }
        self.validation = {
            "run_id": self.run_id, "run_directory": self.recorded_run,
            "validation_status": "passed", "approved_for_delivery": False,
            "model_sha256": registrar.recorded_identity(self.run / registrar.MODEL)["sha256"],
            "model_validation": {"model_sha256": registrar.recorded_identity(self.run / registrar.MODEL)["sha256"]},
        }
        self.dump(self.run / "data_validation.json", {"fixture": True})
        self.dump(self.run / "validation_report.json", self.validation)
        self.evaluation = {
            "run_id": self.run_id, "run_directory": self.recorded_run,
            "evaluation_status": "completed", "approved_for_delivery": False,
            "inputs": {name: {"path": self.recorded_run + "/" + name,
                              **registrar.recorded_identity(self.run / name)} for name in
                       ("validation_report.json", registrar.MODEL, "preprocessed/X_test.csv",
                        "preprocessed/y_test.csv", "preprocessed/y_train.csv")},
        }
        self.dump(self.run / "evaluation_report.json", self.evaluation)
        self.gate = {
            "run_id": self.run_id, "gate_status": "passed", "approved_for_delivery": False,
            "checks": [{"name": "fixture_check", "passed": True}],
            "evaluation_report": {"path": self.recorded_run + "/evaluation_report.json",
                                  "sha256": registrar.recorded_identity(self.run / "evaluation_report.json")["sha256"]},
            "criteria": {"source": {"sha256": self.context["configuration_evidence"][
                "src/acceptance_criteria.json"]["sha256"]}},
        }
        self.dump(self.run / "acceptance_report.json", self.gate)
        self.dump(self.run / "metrics.json", {"fixture": True})
        self.receipt = {"schema_name": "staged_pipeline_receipt_v1", "invocation_id": self.invocation, "stages": []}
        for index, name in enumerate(("prepare", "train", "evaluate")):
            command = ["python", "-B", "src/pipeline_stages.py", name]
            if name == "prepare":
                command += ["--raw-dir", self.context["raw_directory"]]
            command += ["--run-dir", self.recorded_run]
            if name == "evaluate":
                command += ["--criteria", "/app/src/acceptance_criteria.json"]
            self.receipt["stages"].append({
                "name": name, "image_id": self.context["image_id"], "command": command,
                "exit_code": 0, "launch_error_type": None,
                "started_at_utc": "2026-11-07T10:0%d:01+00:00" % index,
                "finished_at_utc": "2026-11-07T10:0%d:02+00:00" % index})
        self.dump(self.receipt_path, self.receipt)
        self.provenance = {
            "schema_name": "staged_pipeline_provenance_v1", "invocation": self.context,
            "mlflow_run_id": self.mlflow, "terminal_status": "FINISHED", "gate_status": "passed",
            "approved_for_delivery": False, "stage_receipt_sha256": registrar.recorded_identity(self.receipt_path)["sha256"],
        }
        self.refresh_artifacts()
        for target in (socket.socket,):
            guard = patch.object(target, "connect", side_effect=AssertionError("Network forbidden in offline fixtures."))
            guard.start()
            self.addCleanup(guard.stop)
        for method in ("fit", "predict", "score"):
            guard = patch.object(RandomForestClassifier, method,
                                 side_effect=AssertionError("Estimator execution forbidden in fixtures."))
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")

    def refresh_artifacts(self):
        self.provenance["artifacts"] = {
            "code/src/run_pipeline.py": {"sha256": "5" * 64, "size_bytes": 1},
            **{"outputs/" + name: registrar.recorded_identity(self.run / name) for name in
               (*registrar.PREPARED, registrar.MODEL, *registrar.REPORTS)},
        }
        self.dump(self.provenance_path, self.provenance)

    def arguments(self):
        return dict(repository=self.repo, raw_directory=self.raw, run_directory=self.run,
                    provenance_path=self.provenance_path, receipt_path=self.receipt_path,
                    workspace=self.workspace, invocation_id=self.invocation, run_id=self.run_id,
                    mlflow_run_id=self.mlflow, original_pointers=(self.pointer,))

    def validate(self):
        return registrar.validate_inputs(self.repo, self.raw, self.run, self.provenance_path,
                                         self.receipt_path, self.invocation, self.run_id, self.mlflow)

    def protected(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*")
                if path.is_file() and self.workspace not in path.parents}

    def test_actual_local_dvc_links_opaque_artifacts_and_preserves_inputs(self):
        before = self.protected()
        poisoned = self.root / "existing_global"
        poisoned.mkdir()
        sentinel = poisoned / "config"
        sentinel.write_text('[core]\nremote = forbidden\n[remote "forbidden"]\nurl = https://invalid.invalid\n')
        with patch.dict(os.environ, {"DVC_GLOBAL_CONFIG_DIR": str(poisoned), "DVC_ROOT": str(self.repo),
                                     "DVC_STUDIO_TOKEN": "fixture_value_never_propagated"}):
            manifest = registrar.register_pipeline_run(**self.arguments())
        self.assertEqual(manifest["status"], "registered_locally")
        self.assertEqual(set(manifest["objects"]), {*registrar.PREPARED, registrar.MODEL})
        self.assertEqual(manifest["original_training"], self.context)
        self.assertEqual(manifest["mlflow_run_id"], self.mlflow)
        self.assertEqual(manifest["estimator_configuration_evidence"]["status"], "missing_in_original_validation_report")
        self.assertFalse(manifest["approved_for_delivery"])
        self.assertEqual(manifest["remote_confirmation"], "not_attempted")
        self.assertNotEqual(manifest["registration"]["source_identity"]["sha256"], self.context["source_sha256"]["src/run_pipeline.py"])
        for key, value in manifest["objects"].items():
            self.assertEqual((self.workspace / value["workspace_path"]).read_bytes(), (self.run / key).read_bytes())
            self.assertEqual((self.workspace / value["dvc_object"]["local_cache_path"]).read_bytes(), (self.run / key).read_bytes())
        for relative, content in before.items():
            self.assertEqual((self.root / relative).read_bytes(), content)
        self.assertIn("forbidden", sentinel.read_text())
        self.assertFalse((self.repo / ".dvc").exists())
        self.assertFalse((self.workspace / ".git").exists())
        self.assertEqual(json.loads((self.workspace / "link_manifest.json").read_bytes()), manifest)
        self.assertEqual((self.workspace / "evidence/stage_receipt.json").read_bytes(), self.receipt_path.read_bytes())
        self.assertEqual((self.workspace / "evidence/run_provenance.json").read_bytes(), self.provenance_path.read_bytes())
        for command_path in self.workspace.glob("command_*.json"):
            self.assertEqual(json.loads(command_path.read_bytes())["exit_code"], 0)

    def test_new_report_parameters_use_one_existing_loader_and_preserve_none(self):
        model = RandomForestClassifier(n_jobs=-1)
        model.feature_names_in_, model.n_features_in_, model.classes_ = np.array(pipeline.FEATURES), len(pipeline.FEATURES), np.array([0, 1])
        historical = self.protected()
        with patch.object(pipeline.joblib, "load", return_value=model) as loader:
            validation = pipeline.validate_model_artifact(self.run / registrar.MODEL)
        loader.assert_called_once()
        output = self.root / "new_validation.json"
        pipeline.write_validation_report(output, self.run, {}, validation)
        report = json.loads(output.read_bytes())
        configuration = report["model_validation"]["estimator_configuration"]
        self.assertEqual(configuration["method"], "get_params(deep=False)")
        self.assertEqual(configuration["parameters"], model.get_params(deep=False))
        self.assertIsNone(configuration["parameters"]["random_state"])
        self.assertEqual(configuration["parameters"]["n_jobs"], -1)
        for relative, content in historical.items():
            self.assertEqual((self.root / relative).read_bytes(), content)
        with self.assertRaises(FileExistsError):
            pipeline.write_validation_report(output, self.run, {}, validation)

    def test_json_parameter_serialization_is_explicit(self):
        values = {"none": None, "integer": np.int64(7), "tuple": (1, np.float64(2.5)),
                  "nan": float("nan"), "object": object()}
        safe = pipeline.json_safe_parameters(values)
        self.assertEqual(json.loads(json.dumps(safe, allow_nan=False))["tuple"], [1, 2.5])
        self.assertIsNone(safe["none"])
        self.assertEqual(safe["nan"]["status"], "non_finite")
        self.assertEqual(safe["object"]["status"], "unsupported_type")

    def test_recorded_parameters_are_bound_to_original_report(self):
        configuration = {"method": "get_params(deep=False)", "parameters": {"random_state": None, "n_jobs": -1}}
        self.validation["model_validation"]["estimator_configuration"] = configuration
        self.dump(self.run / "validation_report.json", self.validation)
        self.evaluation["inputs"]["validation_report.json"].update(registrar.recorded_identity(self.run / "validation_report.json"))
        self.dump(self.run / "evaluation_report.json", self.evaluation)
        self.gate["evaluation_report"]["sha256"] = registrar.recorded_identity(self.run / "evaluation_report.json")["sha256"]
        self.dump(self.run / "acceptance_report.json", self.gate)
        self.refresh_artifacts()
        _, parameters, _ = self.validate()
        self.assertEqual(parameters["evidence"], configuration)
        self.assertEqual(parameters["status"], "recorded_in_original_validation_report")

    def test_raw_and_lf_pointer_representations_do_not_claim_data_identity(self):
        observed = registrar.pointer_representation(self.pointer)
        raw = self.pointer.read_bytes()
        self.assertEqual(observed["raw_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(observed["lf_comparison_sha256"], hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
        self.assertNotEqual(observed["raw_sha256"], observed["lf_comparison_sha256"])
        self.assertFalse(observed["data_identity_established_by_pointer_text"])
        self.assertEqual(self.pointer.read_bytes(), raw)

    def test_mismatched_ids_hashes_completion_and_associations_are_rejected(self):
        mutations = (
            lambda p: p.update(mlflow_run_id="9" * 32),
            lambda p: p["invocation"].update(invocation_id="9" * 32),
            lambda p: p.update(stage_receipt_sha256="0" * 64),
            lambda p: p.update(terminal_status="FAILED"),
            lambda p: p.update(gate_status="failed"),
            lambda p: p.update(approved_for_delivery=True),
            lambda p: p["invocation"].update(run_directory="/app/data/runs/other"),
            lambda p: p["artifacts"]["outputs/" + registrar.MODEL].update(sha256="0" * 64),
            lambda p: p["artifacts"]["code/src/run_pipeline.py"].update(sha256="0" * 64),
            lambda p: p["invocation"]["raw_identities"][registrar.RAW[0]].update(sha256="0" * 64),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                candidate = copy.deepcopy(self.provenance)
                mutate(candidate)
                self.dump(self.provenance_path, candidate)
                with self.assertRaises(ValueError):
                    registrar.register_pipeline_run(**self.arguments())
                self.assertFalse(self.workspace.exists())
        self.dump(self.provenance_path, self.provenance)

    def test_report_associations_are_checked_beyond_top_level_hashes(self):
        originals = {name: (self.run / name).read_bytes() for name in registrar.REPORTS}
        cases = (
            ("validation_report.json", lambda report: report.update(run_id="other")),
            ("validation_report.json", lambda report: report.update(model_sha256="0" * 64)),
            ("evaluation_report.json", lambda report: report["inputs"][registrar.MODEL].update(sha256="0" * 64)),
            ("acceptance_report.json", lambda report: report["evaluation_report"].update(sha256="0" * 64)),
            ("acceptance_report.json", lambda report: report["checks"][0].update(passed=False)),
            ("acceptance_report.json", lambda report: report["criteria"]["source"].update(sha256="0" * 64)),
        )
        for index, (name, mutate) in enumerate(cases):
            with self.subTest(case=index):
                for relative, content in originals.items():
                    (self.run / relative).write_bytes(content)
                report = json.loads(originals[name])
                mutate(report)
                self.dump(self.run / name, report)
                self.refresh_artifacts()
                with self.assertRaises(ValueError):
                    registrar.register_pipeline_run(**self.arguments())
                self.assertFalse(self.workspace.exists())

    def test_artifact_tampering_is_rejected_before_workspace_creation(self):
        (self.run / registrar.PREPARED[0]).write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "Actual run artifact differs"):
            registrar.register_pipeline_run(**self.arguments())
        self.assertFalse(self.workspace.exists())

    def test_receipt_stage_failure_is_rejected_even_with_matching_receipt_hash(self):
        self.receipt["stages"][1]["exit_code"] = 1
        self.dump(self.receipt_path, self.receipt)
        self.provenance["stage_receipt_sha256"] = registrar.recorded_identity(self.receipt_path)["sha256"]
        self.dump(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Stage completion"):
            registrar.register_pipeline_run(**self.arguments())
        self.assertFalse(self.workspace.exists())

    def test_existing_or_overlapping_workspace_is_preserved(self):
        self.workspace.mkdir()
        sentinel = self.workspace / "preserved.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "Existing workspace preserved"):
            registrar.register_pipeline_run(**self.arguments())
        self.assertEqual(sentinel.read_bytes(), b"keep")
        arguments = self.arguments()
        arguments["workspace"] = self.repo / "new_workspace"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            registrar.register_pipeline_run(**arguments)
        self.assertFalse(arguments["workspace"].exists())

    def test_windows_username_survives_isolation_without_inheriting_credentials(self):
        with patch.dict(os.environ, {
            "USERNAME": "fixture-user", "DVC_STUDIO_TOKEN": "forbidden",
            "DVC_ROOT": "forbidden", "GIT_DIR": "forbidden", "PYTHONPATH": "forbidden",
        }):
            local = registrar.LocalDvc(self.root / "username_environment_fixture")
        self.assertEqual(local.env["USERNAME"], "fixture-user")
        for name in ("DVC_STUDIO_TOKEN", "DVC_ROOT", "GIT_DIR", "PYTHONPATH"):
            self.assertNotIn(name, local.env)
        with patch.dict(os.environ, local.env, clear=True):
            self.assertEqual(getpass.getuser(), "fixture-user")

    def test_isolation_environment_and_failed_command_have_no_retry(self):
        with patch.dict(os.environ, {"DVC_ROOT": "forbidden", "DVC_STUDIO_TOKEN": "forbidden",
                                     "GIT_DIR": "forbidden", "PYTHONPATH": "forbidden"}):
            local = registrar.LocalDvc(self.root / "environment_fixture")
        for name in ("DVC_ROOT", "DVC_STUDIO_TOKEN", "GIT_DIR", "PYTHONPATH"):
            self.assertNotIn(name, local.env)
        for name in ("DVC_GLOBAL_CONFIG_DIR", "DVC_SYSTEM_CONFIG_DIR", "DVC_SITE_CACHE_DIR"):
            self.assertTrue(Path(local.env[name]).is_relative_to(self.root))
        with patch.object(registrar.LocalDvc, "command", side_effect=RuntimeError("fixture command failure")) as command:
            with self.assertRaisesRegex(RuntimeError, "fixture command failure"):
                registrar.register_pipeline_run(**self.arguments())
        command.assert_called_once_with("--version")
        failure = json.loads((self.workspace / "registration_failure.json").read_bytes())
        self.assertEqual(failure["error"], "fixture command failure")
        self.assertFalse((self.workspace / "link_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
