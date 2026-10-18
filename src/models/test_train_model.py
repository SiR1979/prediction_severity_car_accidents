"""Check direct reference training orchestration without fitting or serializing a model."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.models import train_model as training


class TrainModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="training-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "prepared"
        self.data.mkdir()
        self.destination = self.root / "new-output/model.bin"
        self.features_path = self.data / "X_train.csv"
        self.target_path = self.data / "y_train.csv"
        self.features_path.write_text("speed,lane_count,indicator\n1.5,2,0\n3.5,1,1\n2.5,3,0\n", encoding="utf-8")
        self.target_path.write_text("grav\n0\n1\n0\n", encoding="utf-8")
        self.artifact = b"opaque serialized fixture\x00\xff"
        self.model = Mock(spec=["fit"])
        self.enterContext(patch.object(training.ensemble.RandomForestClassifier, "fit",
                                      side_effect=AssertionError("Real fitting forbidden")))
        self.estimator = self.enterContext(patch.object(training.ensemble, "RandomForestClassifier",
                                                       return_value=self.model))
        self.serializer = self.enterContext(patch.object(training.joblib, "dump", side_effect=self.write_opaque))
        self.enterContext(patch.object(training.joblib, "load", side_effect=AssertionError("Model loading forbidden")))
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Subprocess forbidden")))
        self.enterContext(patch("subprocess.run", side_effect=AssertionError("Subprocess forbidden")))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden")))

    def write_opaque(self, model, stream):
        self.assertIs(model, self.model)
        self.assertEqual(Path(stream.name), self.destination)
        self.assertEqual(stream.mode, "xb")
        stream.write(self.artifact)

    def input_hashes(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (self.features_path, self.target_path) if path.exists()}

    def train(self):
        return training.train_model(self.data, self.destination)

    def test_order_values_flattened_target_and_reference_configuration_reach_the_mock(self):
        """The direct function preserves CSV feature order and configures only the reference estimator."""
        before = self.input_hashes()
        result = self.train()
        self.assertEqual(result, self.destination)
        self.estimator.assert_called_once_with(n_jobs=-1)
        self.model.fit.assert_called_once()
        features, target = self.model.fit.call_args.args
        expected = pd.DataFrame({"speed": [1.5, 3.5, 2.5], "lane_count": [2, 1, 3], "indicator": [0, 1, 0]})
        pd.testing.assert_frame_equal(features, expected)
        np.testing.assert_array_equal(target, np.array([0, 1, 0]))
        self.assertEqual(target.shape, (3,))
        self.serializer.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), self.artifact)
        self.assertEqual(self.input_hashes(), before)

    def test_empty_features_and_multicolumn_target_are_rejected_before_estimator_creation(self):
        """Empty feature frames and multiple target columns cannot reach even the estimator mock."""
        original_features = self.features_path.read_bytes()
        original_target = self.target_path.read_bytes()
        for invalid_features in (True, False):
            with self.subTest(empty_features=invalid_features):
                self.features_path.write_bytes(b"speed,lane_count,indicator\n" if invalid_features else original_features)
                self.target_path.write_bytes(original_target if invalid_features else b"grav,extra\n0,1\n1,1\n0,1\n")
                before = self.input_hashes()
                with self.assertRaisesRegex(ValueError, "Training features or target structure are invalid"):
                    self.train()
                self.assertFalse(self.destination.parent.exists())
                self.assertEqual(self.input_hashes(), before)
        self.estimator.assert_not_called()
        self.model.fit.assert_not_called()
        self.serializer.assert_not_called()

    def test_row_count_mismatch_and_header_only_target_are_rejected(self):
        """Unequal feature and target counts are rejected without creating output state."""
        for content in (b"grav\n0\n1\n", b"grav\n"):
            with self.subTest(target=content):
                self.target_path.write_bytes(content)
                before = self.input_hashes()
                with self.assertRaisesRegex(ValueError, "Training feature and target row counts differ"):
                    self.train()
                self.assertFalse(self.destination.parent.exists())
                self.assertEqual(self.input_hashes(), before)
        self.estimator.assert_not_called()
        self.serializer.assert_not_called()

    def test_missing_and_zero_byte_csv_inputs_preserve_available_bytes(self):
        """Missing or unparseable input CSVs stop before estimator creation or output writes."""
        for path in (self.features_path, self.target_path):
            original = path.read_bytes()
            for empty in (False, True):
                with self.subTest(input=path.name, empty=empty):
                    path.unlink()
                    if empty:
                        path.write_bytes(b"")
                    before = self.input_hashes()
                    error = pd.errors.EmptyDataError if empty else FileNotFoundError
                    with self.assertRaises(error):
                        self.train()
                    self.assertFalse(self.destination.parent.exists())
                    self.assertEqual(self.input_hashes(), before)
                    path.write_bytes(original)
        self.estimator.assert_not_called()
        self.serializer.assert_not_called()

    def test_existing_output_file_or_directory_is_preserved_before_reading_inputs(self):
        """Existing files and directories are rejected without parsing data or replacing bytes."""
        for directory in (False, True):
            with self.subTest(directory=directory):
                self.destination.parent.mkdir(exist_ok=True)
                if directory:
                    self.destination.mkdir()
                    sentinel = self.destination / "existing.bin"
                else:
                    sentinel = self.destination
                sentinel.write_bytes(b"preserve existing output")
                before = self.input_hashes()
                with patch.object(training.pd, "read_csv", side_effect=AssertionError("Unexpected parsing")) as reader:
                    with self.assertRaises(FileExistsError):
                        self.train()
                    reader.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b"preserve existing output")
                self.assertEqual(self.input_hashes(), before)
                sentinel.unlink()
                if directory:
                    self.destination.rmdir()
        self.estimator.assert_not_called()
        self.serializer.assert_not_called()

    def test_exclusive_output_creation_preserves_a_file_created_during_fit(self):
        """An output appearing after the initial guard survives the exclusive-open race."""
        before = self.input_hashes()

        def occupy_output(*arguments):
            self.destination.parent.mkdir()
            self.destination.write_bytes(b"concurrent fixture output")

        self.model.fit.side_effect = occupy_output
        with self.assertRaises(FileExistsError):
            self.train()
        self.assertEqual(self.destination.read_bytes(), b"concurrent fixture output")
        self.model.fit.assert_called_once()
        self.serializer.assert_not_called()
        self.assertEqual(self.input_hashes(), before)

    def test_fit_failure_preserves_inputs_without_creating_the_output_parent(self):
        """An estimator exception propagates after one mocked fit and before any artifact write."""
        before = self.input_hashes()
        self.model.fit.side_effect = RuntimeError("simulated fit failure")
        with self.assertRaisesRegex(RuntimeError, "simulated fit failure"):
            self.train()
        self.estimator.assert_called_once_with(n_jobs=-1)
        self.model.fit.assert_called_once()
        self.serializer.assert_not_called()
        self.assertFalse(self.destination.parent.exists())
        self.assertEqual(self.input_hashes(), before)

    def test_serialization_failure_preserves_partial_opaque_output_without_retry(self):
        """A failing serializer leaves its partial bytes and input files unchanged."""
        before = self.input_hashes()

        def fail_after_partial_write(model, stream):
            self.assertIs(model, self.model)
            self.assertEqual(stream.mode, "xb")
            stream.write(b"partial opaque artifact")
            raise OSError("simulated serialization failure")

        self.serializer.side_effect = fail_after_partial_write
        with self.assertRaisesRegex(OSError, "simulated serialization failure"):
            self.train()
        self.model.fit.assert_called_once()
        self.serializer.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), b"partial opaque artifact")
        self.assertEqual(self.input_hashes(), before)

    def test_serialization_failure_before_writing_preserves_the_exclusively_created_file(self):
        """A serializer failing immediately leaves an empty artifact instead of cleaning or retrying."""
        before = self.input_hashes()
        self.serializer.side_effect = RuntimeError("simulated immediate serialization failure")
        with self.assertRaisesRegex(RuntimeError, "simulated immediate serialization failure"):
            self.train()
        self.assertTrue(self.destination.is_file())
        self.assertEqual(self.destination.read_bytes(), b"")
        self.model.fit.assert_called_once()
        self.serializer.assert_called_once()
        self.assertEqual(self.input_hashes(), before)


if __name__ == "__main__":
    unittest.main()
