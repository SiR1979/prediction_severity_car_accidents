"""One tracking suite: isolated lifecycle doubles and an opaque local HTTP run.

No real preprocessing, estimator fitting or deserialisation occurs. Descriptive
evaluation uses controlled prediction substitutes and isolated prepared files.
The real-server case writes only a new, clearly labelled test experiment.
"""

import base64
import copy
import json
import os
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import run_pipeline as pipeline
from tracking.track_pipeline_run import (
    CONFIG_FILES, SOURCE_FILES, PipelineTracker, TrackingFailure, file_identity,
    json_bytes, sha256,
)


OPAQUE_MODEL = b"OPAQUE-TRACKING-FIXTURE\x00" * 4096


class FakeClient:
    def __init__(self):
        self.calls = []
        self.artifacts = {}
        self.current = None
        self.fail_route = None
        self.ambiguous_create = False
        self.bad_download = False
        self.bad_readback = False
        self.bad_metric_readback = False
        self.fail_failure_update = False
        self.duplicate = False

    def post(self, route, payload):
        self.calls.append((route, copy.deepcopy(payload)))
        if route == self.fail_route:
            raise OSError("Injected metadata failure.")
        if route == "experiments/search":
            return {"experiments": []}
        if route == "experiments/create":
            return {"experiment_id": "1"}
        if route == "runs/search":
            return {"runs": [{"info": {"run_id": "b" * 32}}] if self.duplicate else []}
        if route == "runs/create":
            self.current = {"info": {"run_id": uuid.uuid4().hex, "status": "RUNNING", "lifecycle_stage": "active", "experiment_id": "1",
                                     "start_time": payload["start_time"], "artifact_uri": "mlflow-artifacts:/fixture/artifacts"},
                            "data": {"tags": copy.deepcopy(payload["tags"]), "params": [], "metrics": []}}
            if self.ambiguous_create:
                raise TimeoutError("Create response was lost.")
            return {"run": copy.deepcopy(self.current)}
        if route == "runs/log-batch":
            for name in ("params", "metrics"):
                values = {item["key"]: item for item in self.current["data"][name]}
                values.update({item["key"]: copy.deepcopy(item) for item in payload.get(name, [])})
                self.current["data"][name] = list(values.values())
        elif route == "runs/set-tag":
            tags = {item["key"]: item["value"] for item in self.current["data"]["tags"]}
            tags[payload["key"]] = payload["value"]
            self.current["data"]["tags"] = [{"key": key, "value": value} for key, value in tags.items()]
        elif route == "runs/update":
            if self.fail_failure_update and payload["status"] == "FAILED":
                raise OSError("Failure update unavailable.")
            self.current["info"].update(status=payload["status"], end_time=payload["end_time"])
        return {}

    def run(self, _run_id):
        observed = copy.deepcopy(self.current)
        if self.bad_readback and observed["info"]["status"] == "FINISHED":
            observed["data"]["tags"] = []
        if self.bad_metric_readback and observed["info"]["status"] == "FINISHED":
            observed["data"]["metrics"] = []
        return observed

    def upload(self, _run, name, path, expected_hash):
        self.calls.append(("upload", name))
        if self.fail_route == "upload":
            raise OSError("Upload unavailable.")
        content = Path(path).read_bytes()
        if sha256(content) != expected_hash:
            raise TrackingFailure("Upload source differs.")
        self.artifacts[name] = content

    def download_hash(self, _run, name, expected_size):
        content = self.artifacts[name]
        if len(content) != expected_size:
            raise TrackingFailure("Artifact size differs.")
        return "0" * 64 if self.bad_download else sha256(content)


class TrackingPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="new-run-tracking-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "data/raw"
        self.raw.mkdir(parents=True)
        self.run = self.root / "data/runs/fixture_run"
        self.run.parent.mkdir(parents=True)
        for name in pipeline.RAW_FILENAMES:
            (self.raw / name).write_bytes(b"OPAQUE-RAW-FIXTURE\n")
        self.source_root = Path(pipeline.__file__).resolve().parent.parent
        configuration = {}
        for relative in CONFIG_FILES:
            content = ("OPAQUE-CONFIGURATION-FIXTURE " + relative).encode()
            configuration[relative] = {"sha256": sha256(content), "content_base64": base64.b64encode(content).decode()}
        self.context = {"schema_name": "reference_pipeline_invocation_v1", "evidence_scope": "opaque_test_fixture",
                        "invocation_id": uuid.uuid4().hex, "observed_at_utc": pipeline.utc_now(),
                        "git_head": "a" * 40, "worktree_status": ["opaque test fixture"], "image_id": "sha256:" + "a" * 64,
                        "source_sha256": {relative: file_identity(self.source_root / relative)["sha256"] for relative in SOURCE_FILES},
                        "configuration_evidence": configuration, "raw_directory": str(self.raw), "run_directory": str(self.run)}
        self.context_path = self.root / "invocation_context.json"
        self.write_context()
        self.requirements = self.root / "requirements-resolved.txt"
        self.requirements.write_bytes(b"opaque-fixture==0\n")
        self.versioning = self.root / "versioning"
        self.versioning.mkdir()
        pointer = b"wdir: ..\nouts:\n- path: data/raw\n  md5: " + b"a" * 32 + b".dir\n"
        (self.versioning / "raw_2021.dvc").write_bytes(pointer)
        manifest = {"dvc_version": "3.67.1", "registration_baseline_git_head": "b" * 40,
                    "metadata_file_sha256": {"versioning/raw_2021.dvc": sha256(pointer)},
                    "source_file_sha256": {"data/raw/" + name: file_identity(self.raw / name) for name in pipeline.RAW_FILENAMES}}
        (self.versioning / "reference_manifest.json").write_bytes(json_bytes(manifest))
        self.model = RandomForestClassifier(n_jobs=-1)
        self.model.feature_names_in_ = np.array(pipeline.FEATURES)
        self.model.n_features_in_ = 28
        self.model.classes_ = np.array([0, 1])
        self.model.predict = Mock(return_value=np.array([0, 1]))
        self.process = self.start_patch(pipeline, "process_data", side_effect=self.prepare)
        self.train = self.start_patch(pipeline, "train_model", side_effect=self.produce)
        self.loader = self.start_patch(pipeline.joblib, "load", return_value=self.model)
        for method in ("fit", "predict", "predict_proba", "score"):
            self.start_patch(RandomForestClassifier, method, side_effect=AssertionError("Real fitting/inference is forbidden."))
        self.client = FakeClient()
        self.tracker = self.make_tracker()

    def start_patch(self, owner, name, **kwargs):
        replacement = patch.object(owner, name, **kwargs)
        result = replacement.start()
        self.addCleanup(replacement.stop)
        return result

    def write_context(self):
        self.context_path.write_bytes(json_bytes(self.context))

    def make_tracker(self, *, real=False, experiment=None):
        uri = os.environ.get("PIPELINE_TRACKING_TEST_URI", "http://mlflow:5000") if real else "http://localhost:5000"
        kwargs = {"storage_root": self.root, "versioning_directory": self.versioning, "resolved_requirements": self.requirements}
        if experiment:
            kwargs["experiment"] = experiment
        tracker = PipelineTracker(uri, self.context_path, **kwargs)
        if not real:
            tracker.client = self.client
            tracker.check_version = Mock(return_value=None)
        return tracker

    def prepare(self, *arguments):
        directory = arguments[-1]
        directory.mkdir()
        for partition in ("train", "test"):
            pd.DataFrame(np.ones((2, 28)), columns=pipeline.FEATURES).to_csv(directory / f"X_{partition}.csv", index=False)
            pd.DataFrame({"grav": [0, 1]}).to_csv(directory / f"y_{partition}.csv", index=False)

    def produce(self, _directory, destination):
        destination.parent.mkdir()
        with destination.open("xb") as stream:
            stream.write(OPAQUE_MODEL)
        return destination

    def status(self):
        return json.loads((self.run / "run_status.json").read_bytes())

    def assert_failure(self, expected_type):
        record = self.status()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], expected_type)
        self.assertIs(record["approved_for_delivery"], False)
        self.assertEqual(record["tracking"]["status"], "failed")
        return record

    def test_success_links_provenance_and_all_artifacts(self):
        pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        status = self.status()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["tracking"]["remote_status"], "FINISHED")
        provenance = json.loads(self.client.artifacts["pipeline/run_provenance.json"])
        manifest = json.loads(self.client.artifacts["pipeline/artifact_manifest.json"])
        self.assertEqual(provenance["mlflow_run_id"], status["tracking"]["mlflow_run_id"])
        self.assertEqual(provenance["model_sha256"], sha256(OPAQUE_MODEL))
        self.assertEqual(provenance["executed_source_sha256"], self.context["source_sha256"])
        self.assertEqual(provenance["dvc_input_reference"]["association_status"], "matched")
        self.assertEqual(provenance["configuration"]["features"], list(pipeline.FEATURES))
        self.assertIs(provenance["configuration"]["forest_random_state"], None)
        self.assertFalse(provenance["approved_for_delivery"])
        self.assertTrue(provenance["evaluation_performed"])
        self.assertEqual(provenance["model_quality_approval"], "NOT_ESTABLISHED")
        self.assertNotIn("pipeline/artifact_manifest.json", manifest["artifacts"])
        for name, identity in manifest["artifacts"].items():
            self.assertEqual(sha256(self.client.artifacts[name]), identity["sha256"])
            self.assertEqual(len(self.client.artifacts[name]), identity["size_bytes"])
        report = json.loads(self.client.artifacts["pipeline/validation_report.json"])
        tags = {item["key"]: item["value"] for item in self.client.current["data"]["tags"]}
        model_hash = manifest["artifacts"]["pipeline/model/trained_model.joblib"]["sha256"]
        self.assertEqual(report["model_sha256"], model_hash)
        self.assertEqual(report["model_validation"]["model_sha256"], model_hash)
        self.assertEqual(manifest["model_sha256"], model_hash)
        self.assertEqual(tags["model_sha256"], model_hash)
        self.assertEqual(provenance["technical_report_sha256"], manifest["artifacts"]["pipeline/validation_report.json"]["sha256"])
        self.assertEqual(tags["technical_report_sha256"], provenance["technical_report_sha256"])
        self.assertEqual(tags["provenance_sha256"], manifest["artifacts"]["pipeline/run_provenance.json"]["sha256"])
        self.assertEqual(tags["artifact_manifest_sha256"], sha256(self.client.artifacts["pipeline/artifact_manifest.json"]))
        self.assertEqual(status["tracking"]["artifact_manifest_sha256"], tags["artifact_manifest_sha256"])
        evaluation = json.loads(self.client.artifacts["pipeline/evaluation_report.json"])
        link = provenance["evaluation"]
        self.assertEqual(link["metrics"], evaluation["metrics"])
        self.assertEqual(link["training_majority_baseline"], evaluation["training_majority_baseline"])
        self.assertEqual(link["inputs"], evaluation["inputs"])
        self.assertEqual(link["sha256"], sha256(self.client.artifacts[link["artifact"]]))
        self.assertEqual(manifest["evaluation_report_sha256"], link["sha256"])
        self.assertEqual(tags["evaluation_report_sha256"], link["sha256"])
        self.assertEqual(tags["evaluation_performed"], "true")
        self.assertEqual(status["tracking"]["evaluation_report_sha256"], link["sha256"])
        self.assertEqual(status["evaluation_status"], "completed")
        for relative, observed in link["inputs"].items():
            self.assertEqual({key: observed[key] for key in ("sha256", "size_bytes")}, manifest["artifacts"]["pipeline/" + relative])
        metric_batches = [payload for route, payload in self.client.calls if route == "runs/log-batch" and payload.get("metrics")]
        self.assertEqual(len(metric_batches), 1)
        self.assertEqual(metric_batches[0]["run_id"], provenance["mlflow_run_id"])
        recorded_metrics = {item["key"]: item["value"] for item in metric_batches[0]["metrics"]}
        self.assertEqual(recorded_metrics, {prefix + "_" + key: value for prefix, values in (
            ("evaluation", evaluation["metrics"]), ("training_majority", evaluation["training_majority_baseline"]["metrics"])
        ) for key, value in values.items()})
        phases = [payload["value"] for route, payload in self.client.calls if route == "runs/set-tag" and payload["key"] == "pipeline_phase"]
        self.assertEqual(phases, ["preprocessing", "data_validation", "training", "model_validation", "report_writing", "evaluation", "artifact_recording", "tracking_finish"])

    def test_artifact_changes_after_manifest_prevent_completion(self):
        cases = {"report": "validation_report.json", "evaluation": "evaluation_report.json", "model": "model/trained_model.joblib",
                 "prepared": "preprocessed/X_train.csv", "provenance": "tracking_evidence/run_provenance.json",
                 "manifest": "tracking_evidence/artifact_manifest.json"}
        for case, relative in cases.items():
            with self.subTest(artifact=case):
                self.run = self.root / "data/runs" / case
                self.context.update(run_directory=str(self.run), invocation_id=uuid.uuid4().hex)
                self.write_context()
                self.client = FakeClient()
                tracker = self.make_tracker()
                original_set_tags = tracker.set_tags
                changed = []

                def change_after_manifest(values):
                    original_set_tags(values)
                    if "artifact_manifest_sha256" in values:
                        path = self.run / relative
                        content = path.read_bytes()
                        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
                        changed.append(path)

                with patch.object(tracker, "set_tags", side_effect=change_after_manifest):
                    with self.assertRaisesRegex(TrackingFailure, "Local artifact changed before recording"):
                        pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
                self.assertEqual(changed, [self.run / relative])
                self.assertTrue(changed[0].is_file())
                self.assert_failure("TrackingFailure")
                self.assertEqual(self.client.current["info"]["status"], "FAILED")
                self.assertFalse(any(route == "runs/update" and payload["status"] == "FINISHED"
                                     for route, payload in self.client.calls))

    def test_report_model_linkage_uses_captured_identity(self):
        for field in ("model_sha256", "model_path", "validated_sha256", "validated_path"):
            with self.subTest(field=field):
                self.run = self.root / "data/runs" / field
                self.context.update(run_directory=str(self.run), invocation_id=uuid.uuid4().hex)
                self.write_context()
                self.client = FakeClient()
                tracker = self.make_tracker()
                original_finish = tracker.finish

                def inconsistent_report(report_path, model_path, evaluation_path):
                    report = json.loads(report_path.read_bytes())
                    if field.startswith("validated_"):
                        key = "model_sha256" if field == "validated_sha256" else "model_path"
                        report["model_validation"][key] = "0" * 64 if key == "model_sha256" else "different-model.joblib"
                    else:
                        report[field] = "0" * 64 if field == "model_sha256" else "different-model.joblib"
                    report_path.write_bytes(json_bytes(report))
                    original_finish(report_path, model_path, evaluation_path)

                with patch.object(tracker, "finish", side_effect=inconsistent_report):
                    with self.assertRaisesRegex(TrackingFailure, "model identity differs"):
                        pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
                self.assert_failure("TrackingFailure")
                self.assertEqual(self.client.current["info"]["status"], "FAILED")
                self.assertFalse(any(route == "upload" for route, _payload in self.client.calls))
                self.assertFalse(any(route == "runs/update" and payload["status"] == "FINISHED"
                                     for route, payload in self.client.calls))

    def test_artifact_change_during_transfer_prevents_completion(self):
        original_upload = self.client.upload

        def change_after_upload(run, name, path, expected_hash):
            original_upload(run, name, path, expected_hash)
            if name == "pipeline/model/trained_model.joblib":
                content = path.read_bytes()
                path.write_bytes(bytes([content[0] ^ 1]) + content[1:])

        with patch.object(self.client, "upload", side_effect=change_after_upload):
            with self.assertRaisesRegex(TrackingFailure, "Local artifact changed during recording"):
                pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("TrackingFailure")
        self.assertEqual(self.client.current["info"]["status"], "FAILED")
        self.assertEqual(self.client.artifacts["pipeline/model/trained_model.joblib"], OPAQUE_MODEL)
        self.assertNotEqual((self.run / "model/trained_model.joblib").read_bytes(), OPAQUE_MODEL)
        self.assertFalse(any(route == "runs/update" and payload["status"] == "FINISHED"
                             for route, payload in self.client.calls))

    def test_context_source_mismatch_stops_before_remote_writes(self):
        self.context["source_sha256"]["src/run_pipeline.py"] = "0" * 64
        self.write_context()
        with self.assertRaises(TrackingFailure):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("TrackingFailure")
        self.assertFalse(self.client.calls)
        self.train.assert_not_called()

    def test_external_metrics_do_not_determine_completion(self):
        original = self.client.run

        def readback(run_id):
            observed = original(run_id)
            observed["data"]["metrics"].append({"key": "external_fixture_metric", "value": 0})
            return observed

        self.client.run = readback
        pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assertEqual(self.status()["status"], "completed")
        self.assertEqual(len(self.client.current["data"]["metrics"]), 6)

    def test_shared_contract_and_launcher_source_inventory(self):
        import evaluate_run
        import reference_contract

        self.assertIs(pipeline.FEATURES, reference_contract.FEATURES)
        self.assertIs(evaluate_run.FEATURES, reference_contract.FEATURES)
        self.assertEqual(pipeline.CONTRACT_VERSION, evaluate_run.CONTRACT_VERSION)
        launcher = (self.source_root / "scripts/run_tracked_pipeline.ps1").read_text(encoding="utf-8")
        capture = launcher.split("$sourceHashes =", 1)[1].split("$configuration =", 1)[0]
        self.assertEqual(tuple(re.findall(r"'(src/[^']+\.py)'", capture)), SOURCE_FILES)

    def test_evaluation_failure_preserves_primary_error_and_technical_artifacts(self):
        error = ValueError("Primary evaluation failure.")
        original_fail = self.tracker.fail
        with patch.object(pipeline, "evaluate_run", side_effect=error) as evaluate, patch.object(self.tracker, "fail", wraps=original_fail) as fail:
            with self.assertRaises(ValueError) as caught:
                pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assertIs(caught.exception, error)
        evaluate.assert_called_once()
        fail.assert_called_once_with(error)
        record = self.assert_failure("ValueError")
        self.assertEqual(record["error"], "Primary evaluation failure.")
        self.assertEqual(record["technical_validation_status"], "passed")
        self.assertEqual(record["evaluation_status"], "failed")
        self.assertEqual(record["tracking"]["failed_phase"], "evaluation")
        self.assertEqual(json.loads((self.run / "validation_report.json").read_bytes())["validation_status"], "passed")
        self.assertEqual((self.run / "model/trained_model.joblib").read_bytes(), OPAQUE_MODEL)
        self.assertFalse((self.run / "evaluation_report.json").exists())
        self.assertFalse(self.client.artifacts)
        self.assertFalse(any(route == "runs/update" and payload["status"] == "FINISHED" for route, payload in self.client.calls))

    def test_failure_recording_error_does_not_replace_evaluation_error(self):
        error = ValueError("Primary evaluation failure.")
        with patch.object(pipeline, "evaluate_run", side_effect=error), patch.object(self.tracker, "fail", side_effect=OSError("Secondary failure recording error.")), self.assertLogs(level="ERROR"):
            with self.assertRaises(ValueError) as caught:
                pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assertIs(caught.exception, error)
        record = self.status()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], str(error))
        self.assertEqual(record["tracking"]["secondary_error_type"], "OSError")

    def test_evaluated_input_change_before_tracking_blocks_completion(self):
        original = pipeline.evaluate_run

        def evaluate_then_change(directory, output):
            result = original(directory, output)
            with (directory / "preprocessed/y_train.csv").open("ab") as stream:
                stream.write(b"0\n")
            return result

        with patch.object(pipeline, "evaluate_run", side_effect=evaluate_then_change):
            with self.assertRaisesRegex(TrackingFailure, "Evaluated input identity differs"):
                pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        record = self.assert_failure("TrackingFailure")
        self.assertEqual(record["technical_validation_status"], "passed")
        self.assertEqual(record["evaluation_status"], "completed")
        self.assertTrue((self.run / "evaluation_report.json").is_file())
        self.assertFalse(self.client.artifacts)

    def test_missing_metric_readback_prevents_workflow_success(self):
        self.client.bad_metric_readback = True
        with self.assertRaisesRegex(TrackingFailure, "Recorded evaluation metrics differ"):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        record = self.assert_failure("TrackingFailure")
        self.assertEqual(record["evaluation_status"], "completed")
        self.assertEqual(self.client.current["info"]["status"], "FAILED")
        self.assertTrue((self.run / "evaluation_report.json").is_file())

    def test_metric_upload_failure_preserves_completed_evaluation_without_retry(self):
        original = self.client.post
        metric_calls = []

        def post(route, payload):
            if route == "runs/log-batch" and payload.get("metrics"):
                metric_calls.append(copy.deepcopy(payload))
                raise OSError("Evaluation metric recording failed.")
            return original(route, payload)

        self.client.post = post
        with self.assertRaisesRegex(OSError, "Evaluation metric recording failed"):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        record = self.assert_failure("OSError")
        self.assertEqual(len(metric_calls), 1)
        self.assertEqual(record["evaluation_status"], "completed")
        self.assertTrue((self.run / "evaluation_report.json").is_file())
        self.assertEqual(self.client.current["info"]["status"], "FAILED")

    def test_invalid_context_and_target_stop_before_remote_writes(self):
        for case in ("schema", "configuration", "target"):
            with self.subTest(case=case):
                self.run = self.root / "data/runs" / ("invalid_" + case)
                self.context["run_directory"] = str(self.run)
                self.context["schema_name"] = "invalid" if case == "schema" else "reference_pipeline_invocation_v1"
                if case == "configuration":
                    self.context["configuration_evidence"]["Dockerfile"]["sha256"] = "0" * 64
                if case == "target":
                    self.run = self.root / "outside_run"
                    self.context["run_directory"] = str(self.run)
                self.write_context()
                tracker = self.make_tracker()
                with self.assertRaises(TrackingFailure):
                    pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
                self.assert_failure("TrackingFailure")
                self.assertFalse(self.client.calls)

    def test_preprocessing_and_training_failures_are_recorded(self):
        for stage in ("preprocessing", "training"):
            with self.subTest(stage=stage):
                self.run = self.root / "data/runs" / stage
                self.context["run_directory"] = str(self.run)
                self.context["invocation_id"] = uuid.uuid4().hex
                self.write_context()
                self.process.side_effect = ValueError("Original processing error.") if stage == "preprocessing" else self.prepare
                self.train.side_effect = ValueError("Original training error.") if stage == "training" else self.produce
                tracker = self.make_tracker()
                with self.assertRaises(ValueError):
                    pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
                self.assertEqual(self.assert_failure("ValueError")["tracking"]["failed_phase"], stage)
                self.assertEqual(self.client.current["info"]["status"], "FAILED")

    def test_metadata_failure_marks_known_run_failed(self):
        self.client.fail_route = "runs/log-batch"
        with self.assertRaises(OSError):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("OSError")
        self.assertEqual(self.client.current["info"]["status"], "FAILED")
        self.train.assert_not_called()

    def test_technical_validation_failures_prevent_artifact_recording(self):
        for stage in ("data_validation", "model_validation"):
            with self.subTest(stage=stage):
                self.run = self.root / "data/runs" / stage
                self.context["run_directory"] = str(self.run)
                self.context["invocation_id"] = uuid.uuid4().hex
                self.write_context()
                validator = "validate_prepared_data" if stage == "data_validation" else "validate_model_artifact"
                with patch.object(pipeline, validator, side_effect=ValueError("Technical contract failed.")), self.assertRaises(ValueError):
                    pipeline.run_pipeline(self.raw, self.run, tracker=self.make_tracker())
                self.assertEqual(self.assert_failure("ValueError")["tracking"]["failed_phase"], stage)
                self.assertEqual(self.client.current["info"]["status"], "FAILED")
                self.assertFalse(self.client.artifacts)

    def test_nonlocal_or_credentialed_tracking_uris_are_rejected(self):
        for uri in ("https://localhost:5000", "http://example.com:5000", "http://user:secret@localhost:5000", "http://localhost:5000/path"):
            with self.subTest(uri=uri), self.assertRaises(RuntimeError):
                PipelineTracker(uri, self.context_path)
        self.assertFalse(self.run.exists())
        self.assertFalse(self.client.calls)

    def test_ambiguous_creation_is_never_repeated(self):
        self.client.ambiguous_create = True
        with self.assertRaises(TimeoutError):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        tracking = self.assert_failure("TimeoutError")["tracking"]
        self.assertIsNone(tracking["mlflow_run_id"])
        self.assertEqual(tracking["remote_status"], "unknown")
        self.assertEqual(sum(route == "runs/create" for route, _ in self.client.calls), 1)
        self.assertFalse(any(route == "runs/update" for route, _ in self.client.calls))

    def test_duplicate_invocation_is_preserved_without_creation(self):
        self.client.duplicate = True
        with self.assertRaises(TrackingFailure):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("TrackingFailure")
        self.assertFalse(any(route in ("runs/create", "runs/update") for route, _ in self.client.calls))

    def test_upload_and_download_failures_preserve_local_bytes(self):
        for case in ("upload", "download"):
            with self.subTest(case=case):
                self.run = self.root / "data/runs" / case
                self.context["run_directory"] = str(self.run)
                self.context["invocation_id"] = uuid.uuid4().hex
                self.write_context()
                self.client.fail_route = "upload" if case == "upload" else None
                self.client.bad_download = case == "download"
                tracker = self.make_tracker()
                expected = OSError if case == "upload" else TrackingFailure
                with self.assertRaises(expected):
                    pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
                self.assert_failure(expected.__name__)
                self.assertEqual((self.run / "model/trained_model.joblib").read_bytes(), OPAQUE_MODEL)
                self.assertEqual(self.client.current["info"]["status"], "FAILED")

    def test_remote_completion_readback_failure_prevents_local_completion(self):
        self.client.bad_readback = True
        with self.assertRaises(TrackingFailure):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("TrackingFailure")
        self.assertEqual(self.client.current["info"]["status"], "FAILED")

    def test_local_completion_write_failure_marks_remote_failed(self):
        original = Path.write_text

        def write_status(path, text, *args, **kwargs):
            if path.name == "run_status.json" and json.loads(text)["status"] == "completed":
                raise OSError("Local completion write unavailable.")
            return original(path, text, *args, **kwargs)

        with patch.object(Path, "write_text", write_status), self.assertRaises(OSError):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("OSError")
        self.assertEqual(self.client.current["info"]["status"], "FAILED")

    def test_secondary_tracking_failure_preserves_primary_error(self):
        self.process.side_effect = ValueError("Original primary error.")
        self.client.fail_route = "runs/set-tag"
        # Fail only failure tags: phase updates must succeed to reach preprocessing.
        original = self.client.post

        def post(route, payload):
            if route == "runs/set-tag" and payload["key"] == "pipeline_phase":
                self.client.fail_route = None
            elif route == "runs/set-tag" and payload["key"].startswith("pipeline_failed"):
                self.client.fail_route = "runs/set-tag"
            return original(route, payload)

        self.client.post = post
        self.client.fail_failure_update = True
        with self.assertRaisesRegex(ValueError, "Original primary error"):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        tracking = self.assert_failure("ValueError")["tracking"]
        self.assertEqual(tracking["remote_status"], "unknown")
        self.assertTrue(tracking["secondary_error_types"])

    def test_unmatched_dvc_reference_is_not_claimed_as_matching(self):
        (self.raw / pipeline.RAW_FILENAMES[0]).write_bytes(b"DIFFERENT OPAQUE INPUT")
        pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        provenance = json.loads(self.client.artifacts["pipeline/run_provenance.json"])
        self.assertEqual(provenance["dvc_input_reference"]["association_status"], "unmatched")

    def test_existing_evidence_is_not_overwritten(self):
        original = self.produce

        def produce(directory, destination):
            result = original(directory, destination)
            evidence = self.run / "tracking_evidence/run_provenance.json"
            evidence.parent.mkdir()
            evidence.write_bytes(b"PRESERVED SENTINEL")
            return result

        self.train.side_effect = produce
        with self.assertRaises(FileExistsError):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assert_failure("FileExistsError")
        self.assertEqual((self.run / "tracking_evidence/run_provenance.json").read_bytes(), b"PRESERVED SENTINEL")

    def test_existing_run_is_rejected_without_tracking_writes(self):
        self.run.mkdir()
        (self.run / "sentinel").write_bytes(b"PRESERVED")
        with self.assertRaises(FileExistsError):
            pipeline.run_pipeline(self.raw, self.run, tracker=self.tracker)
        self.assertFalse(self.client.calls)
        self.assertEqual((self.run / "sentinel").read_bytes(), b"PRESERVED")

    def test_legacy_mode_and_tracking_cli_errors(self):
        pipeline.run_pipeline(self.raw, self.run)
        self.assertEqual(self.status()["status"], "completed")
        self.assertNotIn("tracking", self.status())
        self.assertFalse(self.client.calls)
        arguments = ["run_pipeline.py", "--raw-dir", str(self.raw), "--run-dir", str(self.run)]
        with patch("sys.argv", arguments + ["--tracking-uri", "http://localhost:5000"]), patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
            pipeline.main()
        self.assertEqual(caught.exception.code, 2)
        with patch("sys.argv", arguments + ["--tracking-uri", "http://localhost:5000", "--tracking-context", str(self.context_path)]), patch("tracking.track_pipeline_run.PipelineTracker", return_value=self.tracker), patch.object(pipeline, "run_pipeline", side_effect=TrackingFailure("Tracked failure.")), self.assertLogs(level="ERROR"):
            self.assertEqual(pipeline.main(), 1)

    def test_empty_tracking_uri_stops_before_pipeline_execution(self):
        arguments = ["run_pipeline.py", "--raw-dir", str(self.raw), "--run-dir", str(self.run), "--tracking-uri", ""]
        for context in ([], ["--tracking-context", str(self.context_path)]):
            with self.subTest(context=context):
                with patch("sys.argv", arguments + context), patch("sys.stderr"), patch.object(pipeline, "run_pipeline") as execute, patch("tracking.track_pipeline_run.PipelineTracker") as create_tracker:
                    with self.assertRaises(SystemExit) as caught:
                        pipeline.main()
                self.assertEqual(caught.exception.code, 2)
                execute.assert_not_called()
                create_tracker.assert_not_called()
        self.process.assert_not_called()
        self.train.assert_not_called()
        self.assertFalse(self.client.calls)
        self.assertFalse(self.run.exists())

    def test_omitted_tracking_options_keep_untracked_cli(self):
        arguments = ["run_pipeline.py", "--raw-dir", str(self.raw), "--run-dir", str(self.run)]
        with patch("sys.argv", arguments), patch("builtins.print"), patch.object(pipeline, "run_pipeline", return_value=self.run) as execute, patch("tracking.track_pipeline_run.PipelineTracker") as create_tracker:
            self.assertEqual(pipeline.main(), 0)
        execute.assert_called_once_with(self.raw, self.run)
        create_tracker.assert_not_called()
        self.assertFalse(self.client.calls)

    def test_opaque_round_trip_against_existing_mlflow(self):
        if not os.environ.get("PIPELINE_TRACKING_TEST_URI"):
            self.skipTest("TODO-VERIFY: no existing MLflow test URI supplied.")
        tracker = self.make_tracker(real=True, experiment="car_accident_reference_pipeline_test_" + uuid.uuid4().hex)
        try:
            tracker.check_version()
        except OSError as error:
            self.skipTest("TODO-VERIFY: existing MLflow server unavailable: " + type(error).__name__)
        pipeline.run_pipeline(self.raw, self.run, tracker=tracker)
        self.assertEqual(self.status()["status"], "completed")
        observed = tracker.client.run(tracker.state["mlflow_run_id"])
        self.assertEqual(observed["info"]["status"], "FINISHED")
        tags = {item["key"]: item["value"] for item in observed["data"]["tags"]}
        self.assertEqual(tags["evidence_scope"], "opaque_test_fixture")
        self.assertEqual(tags["model_sha256"], sha256(OPAQUE_MODEL))
        self.assertTrue({"evaluation_accuracy", "training_majority_accuracy"}.issubset(
            {item["key"] for item in observed["data"].get("metrics", [])}))
        print("OPAQUE_HTTP_RUN_ID=" + tracker.state["mlflow_run_id"], flush=True)


if __name__ == "__main__":
    unittest.main()
