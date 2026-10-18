"""Exercise reference preprocessing directly with small semicolon-delimited fixtures."""

import csv
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data import make_dataset as preprocessing


INPUT_NAMES = ("usagers-2021.csv", "caracteristiques-2021.csv", "lieux-2021.csv", "vehicules-2021.csv")
FEATURES = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv", "obsm", "motor",
    "catr", "circ", "surf", "situ", "vma", "jour", "mois", "lum", "dep", "com", "agg_",
    "int", "atm", "col", "lat", "long", "hour", "nb_victim", "nb_vehicules",
]
HEADERS = (
    "Num_Acc id_vehicule num_veh place catu grav sexe an_nais trajet secu1 secu2 secu3 locp actp etatp",
    "Num_Acc jour mois an hrmn lum dep com agg int atm col adr lat long",
    "Num_Acc catr voie v1 v2 circ nbv vosp prof pr pr1 plan lartpc larrout surf infra situ vma",
    "Num_Acc id_vehicule num_veh senc catv obs obsm choc manv motor occutc",
)


class ProcessDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="preprocessing-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "prepared"
        self.paths = [self.raw / name for name in INPUT_NAMES]
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Subprocess forbidden")))
        self.enterContext(patch("subprocess.run", side_effect=AssertionError("Subprocess forbidden")))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("joblib.load", side_effect=AssertionError("Model loading forbidden")))
        self.enterContext(patch("joblib.dump", side_effect=AssertionError("Model writing forbidden")))
        self.enterContext(patch("sklearn.ensemble.RandomForestClassifier.fit",
                                side_effect=AssertionError("Real fitting forbidden")))

    def make_inputs(self, specifications=None):
        """Create complete raw schemas without reading any project dataset."""
        specifications = [{} for _ in range(10)] if specifications is None else specifications
        tables = [[], [], [], []]
        for marker, specification in enumerate(specifications, start=1):
            accident = 202100000000 + marker
            common = {"Num_Acc": accident, "id_vehicule": "vehicle-%s" % marker, "num_veh": "A1"}
            user = {**common, "place": 1, "catu": 1, "grav": 1, "sexe": 1, "an_nais": 1961,
                    "trajet": 1, "secu1": 1, "secu2": 0, "secu3": 0, "locp": 0, "actp": 0, "etatp": 0}
            characteristic = {"Num_Acc": accident, "jour": marker, "mois": 6, "an": 2021,
                              "hrmn": "08:30", "lum": 1, "dep": "2A", "com": "2A001",
                              "agg": 1, "int": 1, "atm": 1, "col": 1, "adr": "fixture street",
                              "lat": "48,50", "long": "2,25"}
            place = {"Num_Acc": accident, "catr": 3, "voie": 1, "v1": 0, "v2": 0, "circ": 1,
                     "nbv": 2, "vosp": 0, "prof": 0, "pr": 0, "pr1": 0, "plan": 0,
                     "lartpc": 0, "larrout": 0, "surf": 1, "infra": 0, "situ": 1, "vma": 50}
            vehicle = {**common, "senc": 1, "catv": 7, "obs": 0, "obsm": 1, "choc": 0,
                       "manv": 0, "motor": 1, "occutc": 0}
            for row, key in ((user, "user"), (characteristic, "characteristic"),
                             (place, "place"), (vehicle, "vehicle")):
                row.update(specification.get(key, {}))
            for table, row in zip(tables, (user, characteristic, place, vehicle)):
                table.append(row)
            for extra in specification.get("extra_users", []):
                tables[0].append({**user, **extra})
            for index, extra in enumerate(specification.get("extra_vehicles", []), start=2):
                tables[3].append({**vehicle, "id_vehicule": "vehicle-%s-%s" % (marker, index),
                                  "num_veh": "B%s" % index, **extra})
        for path, header, rows in zip(self.paths, HEADERS, tables):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=header.split(), delimiter=";", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

    def input_hashes(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths if path.exists()}

    def process(self, output=None):
        return preprocessing.process_data(*self.paths, self.output if output is None else output)

    def read_outputs(self):
        return {name: pd.read_csv(self.output / (name + ".csv"))
                for name in ("X_train", "X_test", "y_train", "y_test")}

    def aligned_rows(self, outputs):
        rows, targets = {}, {}
        for partition in ("train", "test"):
            features, labels = outputs["X_" + partition], outputs["y_" + partition]
            self.assertEqual(list(features.columns), FEATURES)
            self.assertEqual(list(labels.columns), ["grav"])
            self.assertEqual(len(features), len(labels))
            for row, label in zip(features.to_dict("records"), labels["grav"].tolist()):
                marker = int(row["jour"])
                self.assertNotIn(marker, rows)
                rows[marker], targets[marker] = row, label
        return rows, targets

    def test_real_csv_processing_creates_four_aligned_outputs_and_preserves_inputs(self):
        """Real parsing and splitting produce four new files with aligned features and targets."""
        self.make_inputs()
        before = self.input_hashes()
        self.process()
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"})
        outputs = self.read_outputs()
        self.assertEqual((len(outputs["X_train"]), len(outputs["X_test"])), (7, 3))
        rows, targets = self.aligned_rows(outputs)
        self.assertEqual(set(rows), set(range(1, 11)))
        self.assertEqual(targets, {marker: 1 for marker in range(1, 11)})
        self.assertEqual(self.input_hashes(), before)

    def test_representative_category_location_time_and_age_transformations(self):
        """Independent fixture expectations verify category, coordinate, time and age mappings."""
        specifications = [{} for _ in range(10)]
        specifications[0] = {"characteristic": {"atm": 2}}
        specifications[1] = {"vehicle": {"catv": 13},
                             "characteristic": {"dep": "2B", "com": "2B002", "atm": 9, "hrmn": "17:45"}}
        specifications[2] = {"vehicle": {"catv": 18}, "characteristic": {"atm": 8}}
        specifications[3] = {"vehicle": {"catv": 6}}
        self.make_inputs(specifications)
        before = self.input_hashes()
        self.process()
        rows, _ = self.aligned_rows(self.read_outputs())
        self.assertEqual([rows[marker]["catv"] for marker in (1, 2, 3, 4)], [2, 4, 3, 6])
        self.assertEqual([rows[marker]["atm"] for marker in (1, 2, 3)], [1, 0, 0])
        self.assertEqual((rows[1]["dep"], rows[1]["com"]), (201, 201001))
        self.assertEqual((rows[2]["dep"], rows[2]["com"], rows[2]["hour"]), (202, 202002, 17))
        self.assertEqual((rows[1]["lat"], rows[1]["long"], rows[1]["hour"]), (48.5, 2.25, 8))
        self.assertEqual((rows[1]["year_acc"], rows[1]["victim_age"]), (2021, 60))
        self.assertEqual(self.input_hashes(), before)

    def test_accident_aggregation_counts_rows_and_selects_the_inherited_severity_row(self):
        """Aggregation keeps one severity-selected row and counts all fixture users and vehicles."""
        specifications = [{} for _ in range(10)]
        specifications[0] = {"user": {"grav": 4},
                             "extra_users": [{"grav": 3, "place": 9, "sexe": 2, "an_nais": 1991}],
                             "extra_vehicles": [{}]}
        self.make_inputs(specifications)
        before = self.input_hashes()
        self.process()
        rows, targets = self.aligned_rows(self.read_outputs())
        self.assertEqual(len(rows), 10)
        self.assertEqual((rows[1]["nb_victim"], rows[1]["nb_vehicules"]), (2, 2))
        self.assertEqual((rows[1]["place"], rows[1]["sexe"], rows[1]["victim_age"]), (9, 2, 30))
        self.assertEqual(targets[1], 1)
        self.assertEqual(self.input_hashes(), before)

    def test_required_field_filtering_removes_only_the_five_invalid_accidents(self):
        """Missing required categories remove affected rows while valid accidents remain aligned."""
        specifications = [{} for _ in range(10)]
        specifications[0] = {"user": {"secu1": -1}}
        specifications[1] = {"vehicle": {"catv": 0}}
        specifications[2] = {"vehicle": {"obsm": -1}}
        specifications[3] = {"place": {"vma": -1}}
        specifications[4] = {"characteristic": {"atm": -1}}
        self.make_inputs(specifications)
        before = self.input_hashes()
        self.process()
        rows, targets = self.aligned_rows(self.read_outputs())
        self.assertEqual(set(rows), {6, 7, 8, 9, 10})
        self.assertEqual(targets, {marker: 1 for marker in range(6, 11)})
        self.assertEqual(self.input_hashes(), before)

    def test_missing_values_use_training_modes_rather_than_test_or_combined_modes(self):
        """The actual split fills both partitions from training-only values when their modes differ."""
        # Markers 2 and 9 have value 2; markers 1 and 3 have value 4. Others are missing.
        raw_values = {marker: (4 if marker in (1, 3) else 2 if marker in (2, 9) else -1)
                      for marker in range(1, 11)}
        specifications = [{"place": {"surf": value, "circ": value},
                           "characteristic": {"col": value}, "vehicle": {"motor": value}}
                          for value in raw_values.values()]
        self.make_inputs(specifications)
        before = self.input_hashes()
        observed = {}
        actual_split = preprocessing.train_test_split

        def capture_split(features, target, **options):
            self.assertEqual(options, {"test_size": 0.3, "random_state": 42})
            result = actual_split(features, target, **options)
            observed["train"] = result[0]["jour"].astype(int).tolist()
            observed["test"] = result[1]["jour"].astype(int).tolist()
            return result

        with patch.object(preprocessing, "train_test_split", side_effect=capture_split):
            self.process()

        def independent_mode(markers):
            counts = Counter(raw_values[marker] for marker in markers if raw_values[marker] != -1)
            return min(value for value, count in counts.items() if count == max(counts.values()))

        training_mode = independent_mode(observed["train"])
        self.assertEqual(training_mode, 4)
        self.assertEqual(independent_mode(observed["test"]), 2)
        self.assertEqual(independent_mode(raw_values), 2)
        outputs = self.read_outputs()
        self.aligned_rows(outputs)
        for partition in ("train", "test"):
            frame = outputs["X_" + partition]
            for row in frame.to_dict("records"):
                value = raw_values[int(row["jour"])]
                expected = training_mode if value == -1 else value
                for column in ("surf", "circ", "col", "motor"):
                    self.assertEqual(row[column], expected)
        self.assertEqual(self.input_hashes(), before)

    def test_inherited_target_recoding_is_characterised_without_domain_approval(self):
        """Observed final codes are locked as inherited behaviour, without assigning severity labels."""
        specifications = [{} for _ in range(10)]
        for index, original in enumerate((1, 2, 3, 4)):
            specifications[index] = {"user": {"grav": original}}
        self.make_inputs(specifications)
        self.process()
        _, targets = self.aligned_rows(self.read_outputs())
        # Successive recodings map original codes 1, 2, 3, 4 to 1, 1, 1, 0.
        self.assertEqual([targets[marker] for marker in (1, 2, 3, 4)], [1, 1, 1, 0])

    def test_inherited_out_of_range_ages_remain_visible_without_endorsement(self):
        """The ineffective inherited replacement leaves ages 221 and -9 observable in outputs."""
        specifications = [{} for _ in range(10)]
        specifications[0] = {"user": {"an_nais": 1800}}
        specifications[1] = {"user": {"an_nais": 2030}}
        self.make_inputs(specifications)
        self.process()
        rows, _ = self.aligned_rows(self.read_outputs())
        self.assertEqual((rows[1]["victim_age"], rows[2]["victim_age"]), (221, -9))

    def test_each_missing_or_empty_input_is_rejected_before_creating_output(self):
        """Each input guard rejects missing or zero-byte files without producing output."""
        self.make_inputs()
        for index, path in enumerate(self.paths):
            original = path.read_bytes()
            for empty in (False, True):
                with self.subTest(input=path.name, empty=empty):
                    path.unlink()
                    if empty:
                        path.write_bytes(b"")
                    before = self.input_hashes()
                    destination = self.root / ("rejected-%s-%s" % (index, empty))
                    with self.assertRaisesRegex(ValueError, "Required input is missing or empty"):
                        self.process(destination)
                    self.assertFalse(destination.exists())
                    self.assertEqual(self.input_hashes(), before)
                    path.write_bytes(original)

    def test_existing_output_directory_is_preserved_before_csv_parsing(self):
        """An occupied output directory retains its sentinel and prevents parsing or replacement."""
        self.make_inputs()
        before = self.input_hashes()
        self.output.mkdir()
        sentinel = self.output / "existing.bin"
        sentinel.write_bytes(b"preserve existing output")
        with patch.object(preprocessing.pd, "read_csv", side_effect=AssertionError("Unexpected parsing")) as reader:
            with self.assertRaises(FileExistsError):
                self.process()
            reader.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"preserve existing output")
        self.assertEqual(self.input_hashes(), before)

    def test_write_failure_preserves_inputs_completed_output_and_partial_bytes(self):
        """A second-file write failure retains the first CSV and partial bytes without retry."""
        self.make_inputs()
        before = self.input_hashes()
        actual_write = pd.DataFrame.to_csv

        def failing_write(frame, path, *arguments, **options):
            self.assertEqual(options, {"index": False, "mode": "x"})
            if Path(path).name == "X_test.csv":
                with Path(path).open("xb") as stream:
                    stream.write(b"partial CSV fixture")
                raise OSError("simulated output failure")
            return actual_write(frame, path, *arguments, **options)

        with patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=failing_write) as writer:
            with self.assertRaisesRegex(OSError, "simulated output failure"):
                self.process()
            self.assertEqual(writer.call_count, 2)
        self.assertEqual({path.name for path in self.output.iterdir()}, {"X_train.csv", "X_test.csv"})
        completed = pd.read_csv(self.output / "X_train.csv")
        self.assertEqual((len(completed), list(completed.columns)), (7, FEATURES))
        self.assertEqual((self.output / "X_test.csv").read_bytes(), b"partial CSV fixture")
        self.assertEqual(self.input_hashes(), before)


if __name__ == "__main__":
    unittest.main()
