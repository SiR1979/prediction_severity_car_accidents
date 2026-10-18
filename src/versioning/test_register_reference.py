"""Exercise reference registration with opaque files and simulated local commands."""

import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.versioning import register_reference as reference


class ReferenceRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / "repository"
        self.store = self.root / "storage"
        self.repo.mkdir()
        self.store.mkdir()
        self.baseline = "1" * 40
        self.image = "sha256:" + "2" * 64
        self.head = self.baseline
        self.staged = ""
        self.fail_action = None
        self.corrupt_recovery = False
        self.actions = []
        self.files = {}
        for relative in reference.FILES:
            # Matching prepared partitions remain tiny opaque byte sequences.
            label = Path(relative).name if "/preprocessed/" in relative else relative
            self.files[relative] = ("opaque:" + label).encode() + b"\x00\xff"
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.files[relative])
        (self.repo / ".gitignore").write_bytes(b"fixture exclusions\n")
        self.expected = {name: hashlib.sha256(data).hexdigest() for name, data in self.files.items()}
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Real process forbidden")))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.dict(sys.modules, {"dvc": None, "dvc.repo": None}))
        self.process = self.enterContext(patch.object(reference.subprocess, "run", side_effect=self.simulate))

    def simulate(self, argv, *, cwd, **options):
        """Simulate only the command effects required by the fixture contract."""
        work = Path(cwd)
        self.assertTrue(work.is_relative_to(self.root))
        self.assertEqual(options["env"]["DVC_NO_ANALYTICS"], "1")
        self.assertTrue(options["capture_output"])
        self.assertTrue(options["text"])
        if argv[:3] == [sys.executable, "-m", "dvc"]:
            family, arguments = "dvc", argv[3:]
        else:
            self.assertEqual(argv[0], "git")
            self.assertIn("--no-optional-locks", argv)
            family, arguments = "git", argv[argv.index("-C") + 2:]
        self.actions.append((family, *arguments))
        output = ""
        action = arguments[0]
        if family == "git":
            if arguments == ["rev-parse", "HEAD"]:
                output = self.head
            elif arguments == ["diff", "--cached", "--name-only"]:
                output = self.staged if work == self.repo else ""
            elif action == "reset":
                self.assertEqual(set(arguments[2:]), {".dvc/config", ".dvc/.gitignore", ".dvcignore"})
                self.staged = ""
            elif arguments == ["init", "--quiet"]:
                (work / ".git").mkdir()
            else:
                raise AssertionError("Unexpected Git command: " + repr(arguments))
        elif action == self.fail_action:
            (self.store / "objects" / "partial-transfer").write_bytes(b"preserve partial transfer")
            return types.SimpleNamespace(returncode=7, stdout="fixture output", stderr="fixture command error")
        elif arguments == ["--version"]:
            output = reference.DVC_VERSION
        elif arguments == ["init"]:
            (work / ".dvc").mkdir()
            (work / ".dvc/config").write_bytes(b"")
            (work / ".dvc/.gitignore").write_bytes(b"cache\n")
            (work / ".dvcignore").write_bytes(b"fixture ignore\n")
            if work == self.repo:
                self.staged = ".dvc/config\n.dvc/.gitignore\n.dvcignore"
        elif action == "config":
            self.assertIn(arguments[1], {"core.analytics", "core.autostage", "cache.type"})
        elif action == "remote":
            self.assertEqual(arguments, ["remote", "add", "--local", "--default", "reference", str(self.store / "objects")])
        elif action == "commit":
            self.assertEqual(arguments[1:3], ["--force", "--no-relink"])
            for relative in arguments[3:]:
                with (work / relative).open("a", encoding="utf-8") as stream:
                    stream.write("    md5: fixture-object\n")
        elif action == "status":
            output = "{}"
        elif action in {"push", "pull"}:
            self.assertEqual(arguments[1:3], ["--remote", "reference"])
            for relative in self.files:
                source = self.repo / relative if action == "push" else self.store / "objects" / relative
                destination = self.store / "objects" / relative if action == "push" else work / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            if action == "pull" and self.corrupt_recovery:
                (work / reference.FILES[0]).write_bytes(b"corrupt recovery fixture")
        else:
            raise AssertionError("Unexpected DVC command: " + repr(arguments))
        return types.SimpleNamespace(returncode=0, stdout=output, stderr="")

    def register(self, **overrides):
        arguments = dict(repo=self.repo, store=self.store, baseline=self.baseline,
                         image_id=self.image, expected=self.expected, historical=False)
        arguments.update(overrides)
        return reference.register_and_verify(**arguments)

    def tree_bytes(self, directory):
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def assert_sources_preserved(self):
        for relative, expected in self.files.items():
            self.assertEqual((self.repo / relative).read_bytes(), expected)
        self.assertEqual((self.repo / ".gitignore").read_bytes(), b"fixture exclusions\n")

    def test_snapshot_and_digest_match_independent_fixture_hashes(self):
        """Every selected file retains its independently calculated size and hash."""
        expected = {name: {"sha256": self.expected[name], "size_bytes": len(data)}
                    for name, data in self.files.items()}
        self.assertEqual(reference.snapshot(self.repo), expected)
        self.assertEqual(reference.digest(self.repo / reference.FILES[0]), self.expected[reference.FILES[0]])
        self.process.assert_not_called()

    def test_snapshot_rejects_missing_and_empty_files(self):
        """Missing or empty inputs are rejected without modifying other bytes."""
        selected = self.repo / reference.FILES[0]
        original = selected.read_bytes()
        for empty in (False, True):
            with self.subTest(empty=empty):
                selected.unlink()
                if empty:
                    selected.write_bytes(b"")
                before = self.tree_bytes(self.repo)
                with self.assertRaisesRegex(RuntimeError, "empty|missing"):
                    reference.snapshot(self.repo)
                self.assertEqual(self.tree_bytes(self.repo), before)
                selected.write_bytes(original)

    def test_snapshot_rejects_unexpected_files_in_each_selected_directory(self):
        """Unlisted files in selected directory groups remain untouched."""
        for relative in ("data/raw", "data/preprocessed", reference.GROUPS["validated_reference_run"]):
            with self.subTest(directory=relative):
                extra = self.repo / relative / "unexpected.bin"
                extra.write_bytes(b"preserve unexpected bytes")
                before = self.tree_bytes(self.repo)
                with self.assertRaisesRegex(RuntimeError, "unexpected file"):
                    reference.snapshot(self.repo)
                self.assertEqual(self.tree_bytes(self.repo), before)
                extra.unlink()

    def test_snapshot_rejects_linked_inputs_without_following_them(self):
        """A simulated linked input is rejected without requiring Windows link privileges."""
        selected = self.repo / reference.FILES[0]
        with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == selected):
            with self.assertRaisesRegex(RuntimeError, "Linked reference path"):
                reference.snapshot(self.repo)
        self.assert_sources_preserved()

    def test_registration_rejects_overlapping_storage_before_commands(self):
        """Equal, nested and enclosing storage paths cannot reach registration writes."""
        nested = self.repo / "nested-storage"
        nested.mkdir()
        for storage in (self.repo, nested, self.root):
            with self.subTest(storage=storage):
                before = self.tree_bytes(self.root)
                with self.assertRaisesRegex(RuntimeError, "Storage must be separate"):
                    self.register(store=storage)
                self.assertEqual(self.tree_bytes(self.root), before)
        self.process.assert_not_called()

    def test_baseline_and_staged_work_guards_preserve_the_entire_fixture(self):
        """Changed HEAD or existing staged work stops before any DVC operation."""
        for head, staged, message in (("3" * 40, "", "Baseline revision differs"),
                                      (self.baseline, "unrelated.txt", "staged changes")):
            with self.subTest(message=message):
                self.head, self.staged = head, staged
                before = self.tree_bytes(self.root)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.register()
                self.assertEqual(self.tree_bytes(self.root), before)
        self.assertFalse(any(action[0] == "dvc" for action in self.actions))

    def test_existing_registration_targets_and_storage_are_preserved(self):
        """Occupied targets and storage are rejected before registration writes."""
        for relative in (".dvc", ".dvcignore", "versioning", "occupied-store"):
            with self.subTest(target=relative):
                occupied = self.store / "existing.bin" if relative == "occupied-store" else self.repo / relative
                occupied.write_bytes(b"preserve existing target")
                before = self.tree_bytes(self.root)
                with self.assertRaisesRegex(RuntimeError, "Existing DVC target|Existing storage"):
                    self.register()
                self.assertEqual(self.tree_bytes(self.root), before)
                occupied.unlink()
        self.assertFalse(any(action[0] == "dvc" for action in self.actions))

    def test_changed_historical_hash_is_rejected_before_registration_writes(self):
        """An incorrect expected input hash preserves source and storage bytes."""
        expected = dict(self.expected)
        expected[reference.FILES[0]] = "0" * 64
        before = self.tree_bytes(self.root)
        with self.assertRaisesRegex(RuntimeError, "historical bytes differ"):
            self.register(expected=expected)
        self.assertEqual(self.tree_bytes(self.root), before)

    def test_simulated_registration_and_recovery_preserve_sources(self):
        """Actual orchestration writes five pointers and verifies all recovered fixture files."""
        result = self.register()
        recovered = Path(result["restore_directory"])
        self.assertTrue(recovered.is_relative_to(self.store))
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["file_count"], result["pointer_count"]), (len(self.files), 5))
        self.assertFalse(result["approved_for_delivery"])
        self.assertFalse(result["training_performed"])
        for relative, data in self.files.items():
            self.assertEqual((recovered / relative).read_bytes(), data)
        manifest = json.loads((self.repo / "versioning/reference_manifest.json").read_text())
        self.assertEqual(manifest["source_file_sha256"], {
            name: {"sha256": self.expected[name], "size_bytes": len(data)} for name, data in self.files.items()})
        self.assertEqual(manifest["registration_baseline_git_head"], self.baseline)
        self.assertEqual(manifest["original_training_git_revision"], "unknown")
        report = recovered.parent / "validation_report.json"
        self.assertEqual(json.loads(report.read_text()), result)
        self.assertEqual(sum(action[:2] == ("dvc", "push") for action in self.actions), 1)
        self.assertEqual(sum(action[:2] == ("dvc", "pull") for action in self.actions), 1)
        self.assert_sources_preserved()

    def test_recovery_integrity_mismatch_preserves_original_and_partial_state(self):
        """Corrupt recovered bytes stop validation and remain available for inspection."""
        self.corrupt_recovery = True
        with self.assertRaisesRegex(RuntimeError, "Recovered reference bytes differ"):
            self.register()
        recovered = list((self.store / "validation").glob("*/recovered_repository"))
        self.assertEqual(len(recovered), 1)
        self.assertEqual((recovered[0] / reference.FILES[0]).read_bytes(), b"corrupt recovery fixture")
        self.assertTrue((self.repo / "versioning/reference_manifest.json").is_file())
        self.assertFalse((recovered[0].parent / "validation_report.json").exists())
        self.assert_sources_preserved()

    def test_transfer_command_failure_preserves_partial_state_without_retry(self):
        """A failed push keeps its partial store and metadata and never starts recovery."""
        self.fail_action = "push"
        with self.assertRaisesRegex(RuntimeError, "Local operation failed: dvc push"):
            self.register()
        self.assertEqual((self.store / "objects/partial-transfer").read_bytes(), b"preserve partial transfer")
        self.assertTrue((self.repo / "versioning/raw_2021.dvc").is_file())
        self.assertTrue((self.repo / ".dvc").is_dir())
        self.assertEqual(sum(action[:2] == ("dvc", "push") for action in self.actions), 1)
        self.assertFalse(any(action[:2] == ("dvc", "pull") for action in self.actions))
        self.assert_sources_preserved()

    def test_command_failure_retains_both_diagnostic_streams(self):
        """The real command wrapper reports nonzero output without changing fixtures."""
        self.process.side_effect = None
        self.process.return_value = types.SimpleNamespace(returncode=9, stdout="fixture stdout", stderr="fixture stderr")
        before = self.tree_bytes(self.root)
        with self.assertRaisesRegex(RuntimeError, "fixture stdout\nfixture stderr"):
            reference.command(self.repo, "dvc", "status", quiet=True)
        self.assertEqual(self.tree_bytes(self.root), before)
        self.process.assert_called_once()

    def test_partial_initialisation_accepts_only_the_preserved_empty_state(self):
        """Diagnosed empty partial state is accepted; unexpected bytes are never removed."""
        ignore = b"partial fixture ignore\n"
        mutations = (None, ".dvc/config", ".dvc/extra", ".dvcignore",
                     "store-extra", "objects/existing", "validation/existing.json")
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                base = self.root / ("partial-%s" % index)
                repo, store = base / "repo", base / "store"
                (repo / ".dvc/tmp").mkdir(parents=True)
                (repo / ".dvc/config").write_bytes(b"")
                (repo / ".dvcignore").write_bytes(ignore)
                (store / "objects").mkdir(parents=True)
                (store / "validation/empty").mkdir(parents=True)
                if mutation:
                    target = store / ("extra" if mutation == "store-extra" else mutation) if mutation.startswith(("store-", "objects/", "validation/")) else repo / mutation
                    target.write_bytes(b"preserve partial bytes")
                before = self.tree_bytes(base)
                with patch.object(reference, "PARTIAL_IGNORE_SHA256", hashlib.sha256(ignore).hexdigest()):
                    if mutation is None:
                        reference.validate_partial_initialisation(repo, store)
                    else:
                        with self.assertRaises(RuntimeError):
                            reference.validate_partial_initialisation(repo, store)
                self.assertEqual(self.tree_bytes(base), before)
        self.process.assert_not_called()

    def test_partial_initialisation_uses_a_mocked_repository_without_reinitialising(self):
        """Partial completion opens only the mocked DVC boundary and issues no init command."""
        repository_type = MagicMock()
        module = types.ModuleType("dvc.repo")
        module.Repo = repository_type
        with patch.dict(sys.modules, {"dvc": types.ModuleType("dvc"), "dvc.repo": module}):
            reference.initialise(self.repo, partial=True)
        repository_type.assert_called_once_with(str(self.repo))
        repository_type.return_value.__enter__.assert_called_once()
        self.assertEqual(len(self.actions), 3)
        self.assertTrue(all(action[:2] == ("dvc", "config") for action in self.actions))


if __name__ == "__main__":
    unittest.main()
