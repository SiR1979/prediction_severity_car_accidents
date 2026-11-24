"""Check the stage-run recorder with opaque files and a fake MLflow transport.

No stage runs, no model is loaded and no tracking server is contacted.
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.accept_run import accept_run
from src.reference_contract import CONTRACT_VERSION
from src.tracking import track_stage_run as recorder
from src.tracking.track_pipeline_run import TrackingFailure, file_identity, json_bytes
from test_tracking_pipeline import FakeClient

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "manual__2026-11-24T10:00:00.000000+00:00"
COMMIT = "d" * 40


class StageRunTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage-run-tracking-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "data/raw"
        self.raw.mkdir(parents=True)
        for name in recorder.RAW_FILES:
            (self.raw / name).write_bytes(b"opaque raw input " + name.encode() + b"\n")
        self.run = self.root / "data/airflow_runs/manual__2026-11-24T10-00-00.000000-00-00"
        self.run.mkdir(parents=True)
        self.criteria = self.root / "criteria.json"
        self.criteria.write_bytes(json_bytes({"criteria_version": "team-2026-11-01", "status": "team_defined",
            "min_macro_f1": 0.65, "min_balanced_accuracy": 0.65, "min_macro_f1_gain_over_training_majority": 0.2}))
        (self.run / "airflow_run.json").write_bytes(json_bytes({
            "dag_id": "reference_pipeline", "airflow_run_id": RUN_ID, "source_commit": COMMIT,
            "quality_gate": "team", "criteria": str(self.criteria), "tracking_target": "local"}))
        self.client = FakeClient()
        self.client.target = "local"
        self.client.server_version = lambda: "3.17.0"

    def prepare(self):
        for name in ("X_train", "X_test", "y_train", "y_test"):
            path = self.run / "preprocessed" / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"opaque {name}\n".encode())
        (self.run / "data_validation.json").write_bytes(json_bytes({"opaque": True}))

    def train(self):
        model = self.run / "model/trained_model.joblib"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"opaque model bytes; never deserialize\n")
        (self.run / "validation_report.json").write_bytes(json_bytes({
            "run_id": self.run.name, "run_directory": str(self.run), "validation_status": "passed",
            "approved_for_delivery": False, "model_sha256": file_identity(model)["sha256"],
            "model_validation": {"estimator_configuration": {"estimator_type": "opaque.Estimator",
                                 "parameters": {"n_estimators": 100, "random_state": None}}}}))

    def evaluate(self, macro_f1=0.8):
        evaluation = {"schema_name": "reference_run_evaluation_v1", "evaluation_status": "completed",
            "approved_for_delivery": False, "run_id": self.run.name, "run_directory": str(self.run),
            "contract_version": CONTRACT_VERSION,
            "metrics": {"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": macro_f1},
            "training_majority_baseline": {"metrics": {"accuracy": 0.65, "balanced_accuracy": 0.5, "macro_f1": 0.396}},
            "inputs": {name: {"path": str(self.run / name), **file_identity(self.run / name)} for name in
                       ("validation_report.json", "model/trained_model.joblib", "preprocessed/X_test.csv",
                        "preprocessed/y_test.csv", "preprocessed/y_train.csv")}}
        path = self.run / "evaluation_report.json"
        path.write_bytes(json_bytes(evaluation))
        accept_run(path, self.criteria, self.run / "acceptance_report.json", self.run / "metrics.json")

    def record(self, **changes):
        arguments = {"run_directory": self.run, "raw_directory": self.raw, "criteria_path": self.criteria,
                     "orchestrator": "airflow", "orchestrator_run_id": RUN_ID, "source_commit": COMMIT,
                     "uri": "http://mlflow:5000", "client": self.client}
        arguments.update(changes)
        return recorder.record_stage_run(**arguments)

    def tags(self):
        return {item["key"]: item["value"] for item in self.client.current["data"]["tags"]}

    def params(self):
        return {item["key"]: item["value"] for item in self.client.current["data"]["params"]}

    def metrics(self):
        return {item["key"]: item["value"] for item in self.client.current["data"]["metrics"]}

    def test_passed_gate_is_finished_and_findable_by_the_airflow_run_id(self):
        self.prepare(), self.train(), self.evaluate()
        result = self.record()
        self.assertEqual((result["status"], result["remote_status"], result["gate_status"]), ("recorded", "FINISHED", "passed"))
        tags = self.tags()
        self.assertEqual(tags["orchestrator_run_id"], RUN_ID)
        self.assertEqual(tags["airflow_run_id"], RUN_ID)
        self.assertEqual(tags["source_commit"], COMMIT)
        self.assertEqual(tags["approved_for_delivery"], "false")
        self.assertEqual(tags["failed_stage"], "none")
        search = next(payload for route, payload in self.client.calls if route == "runs/search")
        self.assertIn(f"tags.orchestrator_run_id = '{RUN_ID}'", search["filter"])
        self.assertEqual(self.params()["quality_gate"], "team")
        self.assertEqual(self.params()["min_macro_f1"], "0.65")
        self.assertEqual(self.params()["model_random_state"], "null")
        self.assertAlmostEqual(self.metrics()["macro_f1_gain_over_training_majority"], 0.404)
        self.assertEqual(self.metrics()["quality_gate_passed"], 1.0)
        self.assertIn("outputs/model/trained_model.joblib", self.client.artifacts)
        self.assertIn("outputs/preprocessed/X_train.csv", self.client.artifacts)
        self.assertEqual(json.loads((self.run / recorder.RESULT_FILE).read_text())["mlflow_run_id"],
                         self.client.current["info"]["run_id"])

    def test_tags_name_the_exact_data_and_model_versions_of_the_run(self):
        self.prepare(), self.train(), self.evaluate()
        self.record()
        tags = self.tags()
        model = self.run / "model/trained_model.joblib"
        self.assertEqual(tags["model_sha256"], file_identity(model)["sha256"])
        self.assertEqual(tags["model_dvc_md5"], recorder.md5_file(model))
        self.assertEqual(tags["prepared_data_dvc_md5"], recorder.dvc_hash(self.run / "preprocessed")["md5"])
        self.assertTrue(tags["prepared_data_dvc_md5"].endswith(".dir"))
        versions = json.loads(self.client.artifacts["evidence/run_versions.json"])
        self.assertEqual(versions["prepared_data"]["files"]["preprocessed/X_train.csv"],
                         file_identity(self.run / "preprocessed/X_train.csv"))
        self.assertEqual(versions["raw_data"]["dvc"]["md5"], tags["raw_data_dvc_md5"])

    def test_failed_gate_is_recorded_as_failed(self):
        self.prepare(), self.train(), self.evaluate(macro_f1=0.5)
        result = self.record()
        self.assertEqual((result["status"], result["remote_status"], result["gate_status"]), ("recorded", "FAILED", "failed"))
        self.assertEqual(self.tags()["failed_stage"], "evaluate")
        self.assertEqual(self.tags()["model_quality_approval"], "NOT_ESTABLISHED")
        self.assertEqual(self.metrics()["quality_gate_passed"], 0.0)

    def test_failed_train_stage_records_the_prepared_data_and_no_gate(self):
        self.prepare()
        result = self.record()
        self.assertEqual((result["remote_status"], result["failed_stage"], result["gate_status"]),
                         ("FAILED", "train", "not_established"))
        self.assertIn("prepared_data_dvc_md5", self.tags())
        self.assertNotIn("model_dvc_md5", self.tags())
        self.assertEqual(self.metrics(), {})

    def test_changed_model_after_evaluate_is_rejected_before_any_tracking_call(self):
        self.prepare(), self.train(), self.evaluate()
        (self.run / "model/trained_model.joblib").write_bytes(b"other bytes\n")
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertEqual(self.client.calls, [])

    def test_airflow_record_must_belong_to_the_run(self):
        self.prepare(), self.train(), self.evaluate()
        with self.assertRaises(TrackingFailure):
            self.record(orchestrator_run_id="manual__other")
        self.assertEqual(self.client.calls, [])

    def test_an_existing_mlflow_run_for_the_same_airflow_run_is_not_duplicated(self):
        self.prepare(), self.train(), self.evaluate()
        self.client.duplicate = True
        with self.assertRaises(TrackingFailure):
            self.record()
        self.assertNotIn("runs/create", [route for route, _ in self.client.calls])
        self.assertEqual(json.loads((self.run / recorder.RESULT_FILE).read_text())["status"], "failed")

    def test_a_recorded_run_is_not_recorded_again(self):
        self.prepare(), self.train(), self.evaluate()
        self.record()
        with self.assertRaises(TrackingFailure):
            self.record(client=FakeClient())

    def test_upload_failure_marks_the_mlflow_run_failed(self):
        self.prepare(), self.train(), self.evaluate()
        self.client.fail_route = "upload"
        with self.assertRaises(OSError):
            self.record()
        self.assertEqual(self.client.current["info"]["status"], "FAILED")
        self.assertEqual(json.loads((self.run / recorder.RESULT_FILE).read_text())["remote_status"], "FAILED")

    def test_dvc_runs_need_no_airflow_record(self):
        (self.run / "airflow_run.json").unlink()
        self.prepare(), self.train(), self.evaluate()
        result = self.record(orchestrator="dvc", orchestrator_run_id="dvc-repro-1", extra_parameters={"quality_gate": "team"})
        self.assertEqual(result["remote_status"], "FINISHED")
        self.assertNotIn("airflow_run_id", self.tags())
        self.assertEqual(self.tags()["orchestrator"], "dvc")


class DvcHashTests(unittest.TestCase):
    def test_prepared_data_hash_matches_dvc_lock_when_the_dvc_outputs_exist(self):
        directory = ROOT / "data/pipeline/preprocessed"
        lock = ROOT / "dvc.lock"
        if not (directory.is_dir() and lock.is_file()):
            self.skipTest("No local dvc repro outputs.")
        outs = yaml.safe_load(lock.read_text(encoding="utf-8"))["stages"]["prepare"]["outs"]
        expected = next(item for item in outs if item["path"] == "data/pipeline/preprocessed")
        observed = recorder.dvc_hash(directory)
        self.assertEqual((observed["md5"], observed["size"], observed["nfiles"]),
                         (expected["md5"], expected["size"], expected["nfiles"]))


if __name__ == "__main__":
    unittest.main()
