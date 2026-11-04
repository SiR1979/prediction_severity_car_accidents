"""Exercise stage receipts and tracking with opaque files and fake transport."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from src.accept_run import accept_run
from src.reference_contract import CONTRACT_VERSION
from src.tracking import track_staged_run as staged
from src.tracking.track_pipeline_run import TrackingFailure, file_identity, json_bytes
from test_tracking_pipeline import FakeClient


class StagedTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="staged-recording-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.invocation = "a" * 32
        self.run = self.root / "data/runs/opaque_run"
        self.run.mkdir(parents=True)
        self.raw = self.root / "data/raw"
        self.raw.mkdir()
        for name in staged.RAW_FILES:
            (self.raw / name).write_bytes(b"opaque raw input\n")
        for name in staged.SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque source snapshot\n")
        self.criteria = self.root / "src/acceptance_criteria.json"
        self.criteria.write_bytes(json_bytes({"criteria_version": "team-2026-11-01", "status": "team_defined",
            "min_macro_f1": 0.65, "min_balanced_accuracy": 0.65, "min_macro_f1_gain_over_training_majority": 0.2}))
        configurations = {}
        for name in staged.CONFIGURATIONS:
            content = self.criteria.read_bytes() if name == "src/acceptance_criteria.json" else b"opaque configuration\n"
            configurations[name] = {"sha256": staged.sha256(content), "content_base64": base64.b64encode(content).decode()}
        self.requirements = self.root / "requirements-resolved.txt"
        self.requirements.write_bytes(b"opaque-fixture==0\n")
        self.context_path = self.root / "data/run_invocations" / self.invocation / "invocation_context.json"
        self.context_path.parent.mkdir(parents=True)
        self.receipt_path = self.context_path.parent / "stage_receipt.json"
        self.context = {"schema_name": "staged_pipeline_invocation_v1", "evidence_scope": "opaque_test_fixture",
            "invocation_id": self.invocation, "observed_at_utc": "2026-11-03T12:00:00+00:00", "git_head": "b" * 40,
            "image_id": "sha256:" + "c" * 64, "raw_directory": str(self.raw), "run_directory": str(self.run),
            "source_sha256": {name: file_identity(self.root / name)["sha256"] for name in staged.SOURCES},
            "configuration_evidence": configurations,
            "raw_identities": {name: file_identity(self.raw / name) for name in staged.RAW_FILES}}
        self.receipt = {"schema_name": "staged_pipeline_receipt_v1", "invocation_id": self.invocation, "stages": []}
        for index, name in enumerate(staged.STAGES):
            command = ["python", "-B", "src/pipeline_stages.py", name]
            if name == "prepare":
                command += ["--raw-dir", str(self.raw)]
            command += ["--run-dir", str(self.run)]
            if name == "evaluate":
                command += ["--criteria", "/app/src/acceptance_criteria.json"]
            self.receipt["stages"].append({"name": name, "image_id": self.context["image_id"], "command": command,
                "exit_code": 0, "started_at_utc": f"2026-11-03T12:00:{index * 2 + 1:02}+00:00",
                "finished_at_utc": f"2026-11-03T12:00:{index * 2 + 2:02}+00:00"})
        self.client = FakeClient()
        self.client.target = "local"
        self.client.server_version = lambda: "3.17.0"
        self.write_inputs()

    def write_inputs(self):
        self.context_path.write_bytes(json_bytes(self.context))
        self.receipt_path.write_bytes(json_bytes(self.receipt))

    def outputs(self, passed=True):
        for name in staged.OUTPUTS:
            if name.endswith('.csv') or name.endswith('.joblib'):
                path = self.run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"opaque output bytes; never deserialize\n")
        model = self.run / "model/trained_model.joblib"
        report = {"run_id": self.run.name, "run_directory": str(self.run), "validation_status": "passed",
                  "approved_for_delivery": False, "model_sha256": file_identity(model)["sha256"]}
        (self.run / "validation_report.json").write_bytes(json_bytes(report))
        (self.run / "data_validation.json").write_bytes(json_bytes({"opaque": True}))
        evaluation = {"schema_name": "reference_run_evaluation_v1", "evaluation_status": "completed",
            "approved_for_delivery": False, "run_id": self.run.name, "run_directory": str(self.run),
            "contract_version": CONTRACT_VERSION, "metrics": {"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.8 if passed else 0.6},
            "training_majority_baseline": {"metrics": {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.4}},
            "inputs": {name: {"path": str(self.run / name), **file_identity(self.run / name)} for name in
                       ("validation_report.json", "model/trained_model.joblib", "preprocessed/X_test.csv", "preprocessed/y_test.csv", "preprocessed/y_train.csv")}}
        evaluation_path = self.run / "evaluation_report.json"
        evaluation_path.write_bytes(json_bytes(evaluation))
        accept_run(evaluation_path, self.criteria, self.run / "acceptance_report.json", self.run / "metrics.json")

    def record(self):
        return staged.record_staged_run(self.context_path, self.receipt_path, "http://localhost:5000",
            root=self.root, requirements=self.requirements, client=self.client)

    def terminal_updates(self):
        return [payload["status"] for route, payload in self.client.calls if route == "runs/update"]

    def test_success_records_gate_outputs_and_finished(self):
        self.outputs()
        result = self.record()
        self.assertEqual(result["remote_status"], "FINISHED")
        self.assertFalse(result["approved_for_delivery"])
        self.assertEqual(self.terminal_updates(), ["FINISHED"])
        self.assertIn("outputs/acceptance_report.json", self.client.artifacts)
        self.assertIn("outputs/metrics.json", self.client.artifacts)
        self.assertIn("evidence/stage_receipt.json", self.client.artifacts)
        self.assertEqual(len(self.client.current["data"]["metrics"]), 7)

    def test_rejected_gate_is_recorded_as_failed_without_finished_transition(self):
        self.outputs(False)
        self.receipt["stages"][-1]["exit_code"] = 1
        self.write_inputs()
        result = self.record()
        self.assertEqual(result["remote_status"], "FAILED")
        self.assertEqual(result["gate_status"], "failed")
        self.assertEqual(self.terminal_updates(), ["FAILED"])
        self.assertIn("outputs/acceptance_report.json", self.client.artifacts)

    def test_prepare_failure_records_prefix_and_no_successful_gate(self):
        self.receipt["stages"] = self.receipt["stages"][:1]
        self.receipt["stages"][0]["exit_code"] = 2
        self.write_inputs()
        self.assertEqual(self.record()["remote_status"], "FAILED")
        self.assertEqual(self.terminal_updates(), ["FAILED"])

    def test_unknown_container_launch_exit_is_not_fabricated(self):
        self.receipt["stages"] = self.receipt["stages"][:1]
        self.receipt["stages"][0].update(exit_code=None, launch_error_type="IOException")
        self.write_inputs()
        self.assertEqual(self.record()["remote_status"], "FAILED")

    def test_receipt_cannot_continue_after_failed_stage(self):
        self.receipt["stages"][0]["exit_code"] = 2
        self.write_inputs()
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_missing_gate_rejected_before_remote_creation(self):
        self.outputs()
        (self.run / "acceptance_report.json").unlink()
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_altered_criteria_or_sources_are_rejected_before_remote_writes(self):
        self.outputs()
        self.criteria.write_bytes(b"changed\n")
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_artifact_round_trip_failure_retains_failed_result(self):
        self.outputs()
        self.client.bad_download = True
        with self.assertRaises(TrackingFailure):
            self.record()
        result = json.loads((self.run / "staged_tracking_result.json").read_bytes())
        self.assertEqual(result["remote_status"], "FAILED")
        self.assertNotIn("FINISHED", self.terminal_updates())

    def test_ambiguous_creation_is_not_retried(self):
        self.outputs()
        self.client.ambiguous_create = True
        with self.assertRaises(TimeoutError):
            self.record()
        self.assertEqual(self.terminal_updates(), [])
        result = json.loads((self.run / "staged_tracking_result.json").read_bytes())
        self.assertIsNone(result["mlflow_run_id"])
        self.assertEqual(result["remote_status"], "unknown")
        count = len(self.client.calls)
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(len(self.client.calls), count)

    def test_source_change_is_rejected_before_remote_writes(self):
        self.outputs()
        (self.root / staged.SOURCES[0]).write_bytes(b"changed\n")
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_missing_evaluation_input_is_rejected_before_remote_writes(self):
        self.outputs()
        evaluation_path = self.run / "evaluation_report.json"
        evaluation = json.loads(evaluation_path.read_bytes())
        del evaluation["inputs"]["preprocessed/y_train.csv"]
        evaluation_path.write_bytes(json_bytes(evaluation))
        (self.run / "acceptance_report.json").unlink()
        (self.run / "metrics.json").unlink()
        accept_run(evaluation_path, self.criteria, self.run / "acceptance_report.json", self.run / "metrics.json")
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_cli_reports_known_guard_without_printing_untrusted_error_details(self):
        arguments = ["track_staged_run", "--context", "context.json", "--receipt", "receipt.json",
                     "--tracking-uri", "http://localhost:5000"]
        for error, expected in ((TrackingFailure("Runtime criteria differ from launcher observation."),
                                 "Runtime criteria differ from launcher observation."),
                                (OSError("untrusted diagnostic payload"), "STAGED_TRACKING_FAILED: OSError")):
            with self.subTest(error=type(error).__name__), patch("sys.argv", arguments), \
                    patch.object(staged, "record_staged_run", side_effect=error), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(staged.main(), 2)
                self.assertIn(expected, output.getvalue())
                self.assertNotIn("untrusted diagnostic payload", output.getvalue())


if __name__ == "__main__":
    unittest.main()
