"""Check candidate identity, phase evidence and preservation with opaque fixtures."""

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.versioning import register_candidate as candidate


class CandidateRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.work = self.root / "registration"
        self.store = self.root / "store"
        self.work.mkdir()
        self.store.mkdir()
        self.content = b"opaque candidate fixture\x00\xff"
        self.expected = {
            "size_bytes": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "md5": hashlib.md5(self.content, usedforsecurity=False).hexdigest(),
        }
        for name, value in (("SIZE", self.expected["size_bytes"]), ("SHA256", self.expected["sha256"]),
                            ("MD5", self.expected["md5"])):
            self.enterContext(patch.object(candidate, name, value))
        self.source = self.work / candidate.MODEL
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(self.content)
        self.remote_object = self.store / "objects/files/md5" / candidate.MD5[:2] / candidate.MD5[2:]
        self.context = {
            "git_head": candidate.HEAD, "branch": "work/candidate-recovery", "image_id": candidate.IMAGE,
            "registrar_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
            "linked_historical_evidence": {"fixture": "opaque historical linkage"},
            "launcher_sha256": "a" * 64, "evidence_directory": str(self.root / "evidence"),
        }
        self.context_path = self.root / "context.json"
        self.context_path.write_text(json.dumps(self.context), encoding="utf-8")
        self.push_mode = "success"
        self.pull_mode = "success"
        self.actions = []
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("Real process forbidden")))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.dict(sys.modules, {"dvc": None, "dvc.repo": None}))
        self.enterContext(patch.dict(os.environ, {"DVC_SITE_CACHE_DIR": "/work/site-cache"}))
        # Redirect container-only paths before any production function can resolve them.
        self.enterContext(patch.object(candidate, "Path", side_effect=self.sandbox_path))
        self.process = self.enterContext(patch.object(candidate.subprocess, "run", side_effect=self.simulate))

    def sandbox_path(self, value):
        text = str(value)
        if text == "/context.json":
            return self.context_path
        if text == "/store" or text.startswith("/store/"):
            return self.store / text.removeprefix("/store").lstrip("/")
        if text == "/work" or text.startswith("/work/"):
            return self.work / text.removeprefix("/work").lstrip("/")
        return Path(value)

    def pointer_content(self):
        return {"wdir": "..", "outs": [{"path": candidate.MODEL, "md5": candidate.MD5,
                                          "size": candidate.SIZE, "hash": "md5"}]}

    def write_pointer(self, work, content=None):
        path = work / candidate.POINTER
        path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_dump(self.pointer_content() if content is None else content, sort_keys=True).encode()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def prepare_recovery(self, name="recovery"):
        """Build detached metadata independently of the registration method."""
        work = self.root / name
        work.mkdir()
        pointer_sha = self.write_pointer(work)
        manifest = {
            "schema_name": "candidate_model_identity_v1",
            "run_id": candidate.RUN, "invocation_id": candidate.INVOCATION, "mlflow_run_id": candidate.MLFLOW,
            "model": {"relative_path": candidate.MODEL, "size_bytes": len(self.content),
                      "sha256": self.expected["sha256"], "dvc_md5": self.expected["md5"]},
            "pointer": {"relative_path": candidate.POINTER, "sha256": pointer_sha},
            "approved_for_delivery": False,
        }
        (work / candidate.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        self.remote_object.parent.mkdir(parents=True, exist_ok=True)
        self.remote_object.write_bytes(self.content)
        return work

    def simulate(self, argv, *, cwd, **options):
        """Represent DVC effects only inside this test's temporary filesystem."""
        work = Path(cwd)
        self.assertTrue(work.is_relative_to(self.root))
        self.assertEqual(argv[:3], [sys.executable, "-m", "dvc"])
        self.assertEqual(options["env"]["DVC_NO_ANALYTICS"], "1")
        self.assertEqual(options["timeout"], 600)
        self.assertTrue(options["capture_output"])
        self.assertTrue(options["text"])
        arguments = argv[3:]
        self.actions.append((work, tuple(arguments)))
        action = arguments[0]
        output = ""
        if arguments == ["--version"]:
            output = candidate.DVC_VERSION
        elif arguments == ["init", "--no-scm"]:
            (work / ".dvc/cache").mkdir(parents=True)
        elif action == "config":
            self.assertIn(arguments[1], {"core.analytics", "core.autostage", "cache.type", "cache.dir"})
        elif action == "remote":
            self.assertEqual(arguments, ["remote", "add", "--local", "--default", "reference", "/store/objects"])
        elif action == "commit":
            self.assertEqual(arguments, ["commit", "--force", "--no-relink", candidate.POINTER])
            self.write_pointer(work)
        elif action == "status":
            self.assertEqual(arguments, ["status", "--json", candidate.POINTER])
            output = "{}"
        elif action == "push":
            self.assertEqual(arguments, ["push", "--remote", "reference", candidate.POINTER])
            self.remote_object.parent.mkdir(parents=True, exist_ok=True)
            data = (work / candidate.MODEL).read_bytes()
            self.remote_object.write_bytes(data if self.push_mode == "success" else b"partial remote bytes")
            if self.push_mode == "nonzero":
                return types.SimpleNamespace(returncode=9, stdout="fixture stdout", stderr="fixture push error")
        elif action == "pull":
            self.assertEqual(arguments, ["pull", "--remote", "reference", candidate.POINTER])
            destination = work / candidate.MODEL
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.remote_object.read_bytes() if self.pull_mode == "success" else b"partial recovery")
            if self.pull_mode == "timeout":
                raise candidate.subprocess.TimeoutExpired(argv, 600, output="fixture stdout", stderr="fixture timeout")
        else:
            raise AssertionError("Unexpected DVC command: " + repr(arguments))
        return types.SimpleNamespace(returncode=0, stdout=output, stderr="")

    def phase_result(self, work):
        return json.loads((work / "phase_result.json").read_text(encoding="utf-8"))

    def test_identity_matches_independent_hashes_for_opaque_and_empty_files(self):
        """File identities use actual opaque bytes, including the empty-file digest."""
        for data in (self.content, b""):
            with self.subTest(size=len(data)):
                path = self.root / "identity.bin"
                path.write_bytes(data)
                self.assertEqual(candidate.identity(path), {
                    "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()})
        self.process.assert_not_called()

    def test_identity_rejects_missing_directory_and_linked_paths(self):
        """Missing, non-file and simulated linked inputs are rejected without changes."""
        for path in (self.root / "missing.bin", self.store):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "regular file is missing"):
                candidate.identity(path)
        with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == self.source):
            with self.assertRaisesRegex(RuntimeError, "Linked path rejected"):
                candidate.identity(self.source)
        self.assertEqual(self.source.read_bytes(), self.content)

    def test_verify_model_accepts_exact_bytes_and_rejects_each_identity_mismatch(self):
        """Candidate verification checks size and both independent hashes without deserialisation."""
        self.assertEqual(candidate.verify_model(self.source), self.expected)
        for name, invalid in (("SIZE", len(self.content) + 1), ("SHA256", "0" * 64), ("MD5", "0" * 32)):
            with self.subTest(field=name), patch.object(candidate, name, invalid):
                with self.assertRaisesRegex(RuntimeError, "Candidate byte identity differs"):
                    candidate.verify_model(self.source)
        self.assertEqual(self.source.read_bytes(), self.content)

    def test_write_json_creates_new_content_and_preserves_existing_bytes(self):
        """Exclusive JSON creation writes valid content and refuses to replace a file."""
        path = self.work / "exclusive.json"
        candidate.write_json(path, {"status": "fixture", "approved_for_delivery": False})
        before = path.read_bytes()
        self.assertEqual(json.loads(before), {"status": "fixture", "approved_for_delivery": False})
        self.assertTrue(before.endswith(b"\n"))
        with self.assertRaises(FileExistsError):
            candidate.write_json(path, {"replacement": True})
        self.assertEqual(path.read_bytes(), before)

    def test_verify_pointer_accepts_explicit_and_default_md5_hash_kind(self):
        """Valid single-model metadata returns the independently computed pointer hash."""
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                content = self.pointer_content()
                if not explicit:
                    content["outs"][0].pop("hash")
                expected = self.write_pointer(self.work, content)
                self.assertEqual(candidate.verify_pointer(self.work), expected)

    def test_verify_pointer_rejects_scope_path_hash_size_and_unexpected_fields(self):
        """Malformed pointer identities are rejected while their original bytes remain available."""
        cases = []
        for key, value in (("wdir", "."), ("unexpected", True), ("outs", []),
                           ("outs", [self.pointer_content()["outs"][0]] * 2)):
            content = self.pointer_content()
            content[key] = value
            cases.append(content)
        for key, value in (("path", "../outside.bin"), ("md5", "0" * 32),
                           ("size", len(self.content) + 1), ("hash", "sha256"), ("unexpected", True)):
            content = copy.deepcopy(self.pointer_content())
            content["outs"][0][key] = value
            cases.append(content)
        for index, content in enumerate(cases):
            with self.subTest(case=index):
                self.write_pointer(self.work, content)
                pointer = self.work / candidate.POINTER
                before = pointer.read_bytes()
                with self.assertRaises(RuntimeError):
                    candidate.verify_pointer(self.work)
                self.assertEqual(pointer.read_bytes(), before)
        self.process.assert_not_called()

    def test_initialise_rejects_existing_dvc_or_git_workspace_without_commands(self):
        """Previously occupied workspaces keep their sentinel bytes and cannot be replayed."""
        for name in (".dvc", ".git"):
            with self.subTest(name=name):
                work = self.root / ("occupied-" + name[1:])
                (work / name).mkdir(parents=True)
                sentinel = work / name / "existing.bin"
                sentinel.write_bytes(b"preserve workspace")
                with self.assertRaisesRegex(RuntimeError, "Existing DVC workspace|Git repository"):
                    candidate.Phase("registration", work).initialise()
                self.assertEqual(sentinel.read_bytes(), b"preserve workspace")
        self.process.assert_not_called()

    def test_registration_rejects_existing_metadata_or_remote_object(self):
        """Occupied registration metadata and store objects remain unchanged before commands."""
        metadata = self.work / "versioning"
        metadata.mkdir()
        sentinel = metadata / "existing.bin"
        sentinel.write_bytes(b"preserve metadata")
        with self.assertRaisesRegex(RuntimeError, "metadata already exists"):
            candidate.Phase("registration", self.work).registration(self.context)
        self.assertEqual(sentinel.read_bytes(), b"preserve metadata")
        sentinel.unlink()
        metadata.rmdir()
        self.remote_object.parent.mkdir(parents=True)
        self.remote_object.write_bytes(b"preserve store object")
        with self.assertRaisesRegex(RuntimeError, "Existing candidate store object"):
            candidate.Phase("registration", self.work).registration(self.context)
        self.assertEqual(self.remote_object.read_bytes(), b"preserve store object")
        self.process.assert_not_called()

    def test_recovery_rejects_existing_destination_without_commands(self):
        """Recovery refuses to replace existing destination bytes before reading metadata."""
        with self.assertRaisesRegex(RuntimeError, "Recovery destination must be absent"):
            candidate.Phase("recovery", self.work).recovery()
        self.assertEqual(self.source.read_bytes(), self.content)
        self.process.assert_not_called()

    def test_simulated_registration_and_detached_recovery_verify_exact_bytes(self):
        """Actual phase methods register and recover one opaque model with separate evidence."""
        registration = candidate.Phase("registration", self.work)
        registration.run(self.context)
        registered = self.phase_result(self.work)
        self.assertEqual(registered["status"], "passed")
        self.assertEqual(registered["remote_transfer_outcome"], "confirmed_exact_bytes_in_local_store")
        self.assertEqual(registered["dvc_pull_attempts"], 0)
        self.assertEqual(self.remote_object.read_bytes(), self.content)
        manifest_bytes = (self.work / candidate.MANIFEST).read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertEqual(manifest["registration"]["context_sha256"], hashlib.sha256(self.context_path.read_bytes()).hexdigest())
        self.assertFalse(manifest["approved_for_delivery"])
        self.assertFalse(manifest["model_deserialized"])
        recovered = self.root / "detached-recovery"
        (recovered / "versioning").mkdir(parents=True)
        for relative in (candidate.POINTER, candidate.MANIFEST):
            shutil.copyfile(self.work / relative, recovered / relative)
        pointer_bytes = (recovered / candidate.POINTER).read_bytes()
        recovery = candidate.Phase("recovery", recovered)
        recovery.run(self.context)
        result = self.phase_result(recovered)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["restored_identity"], self.expected)
        self.assertEqual(result["dvc_pull_attempts"], 1)
        self.assertFalse(result["source_model_access"])
        self.assertFalse(result["primary_cache_access"])
        self.assertFalse(result["registration_cache_access"])
        self.assertFalse(result["approved_for_delivery"])
        self.assertEqual((recovered / candidate.MODEL).read_bytes(), self.content)
        self.assertEqual((recovered / candidate.POINTER).read_bytes(), pointer_bytes)
        self.assertEqual((recovered / candidate.MANIFEST).read_bytes(), manifest_bytes)
        self.assertEqual(self.source.read_bytes(), self.content)
        self.assertEqual(self.remote_object.read_bytes(), self.content)
        self.assertEqual(sum(args[0] == "pull" for work, args in self.actions if work == recovered), 1)
        self.assertFalse(any(args[0] == "push" for work, args in self.actions if work == recovered))

    def test_run_rejects_changed_context_and_writes_failure_evidence(self):
        """Identity and cache-context mismatches stop before commands and retain failure details."""
        cases = (("git_head", "0" * 40, "Git identity"), ("branch", "other", "Git identity"),
                 ("image_id", "sha256:" + "0" * 64, "image identity"),
                 ("registrar_sha256", "0" * 64, "Mounted registrar"),
                 ("cache", "unexpected", "internal cache path"))
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field):
                work = self.root / ("invalid-context-%s" % index)
                work.mkdir()
                context = dict(self.context)
                if field != "cache":
                    context[field] = value
                with patch.dict(os.environ, {"DVC_SITE_CACHE_DIR": value if field == "cache" else "/work/site-cache"}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        candidate.Phase("registration", work).run(context)
                result = self.phase_result(work)
                self.assertEqual(result["status"], "failed")
                self.assertIn(message, result["primary_error"])
                self.assertEqual(result["command_count"], 0)
                self.assertEqual(result["remote_transfer_outcome"], "not_attempted")
        self.process.assert_not_called()

    def test_recovery_rejects_manifest_mismatches_before_initialising(self):
        """Schema, linkage, model, pointer and approval mismatches preserve detached metadata."""
        for index, field in enumerate(("schema_name", "run_id", "model", "pointer", "approved_for_delivery")):
            with self.subTest(field=field):
                work = self.prepare_recovery("invalid-manifest-%s" % index)
                path = work / candidate.MANIFEST
                manifest = json.loads(path.read_text())
                manifest[field] = True if field == "approved_for_delivery" else "incorrect fixture value"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                before = path.read_bytes()
                pointer_before = (work / candidate.POINTER).read_bytes()
                with self.assertRaises(RuntimeError):
                    candidate.Phase("recovery", work).run(self.context)
                self.assertEqual(self.phase_result(work)["status"], "failed")
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual((work / candidate.POINTER).read_bytes(), pointer_before)
                self.assertFalse((work / ".dvc").exists())
                self.assertFalse((work / candidate.MODEL).exists())
                self.assertEqual(self.remote_object.read_bytes(), self.content)
        self.process.assert_not_called()

    def test_failed_push_keeps_unknown_outcome_and_partial_store_without_retry(self):
        """A nonzero transfer records both streams and leaves partial bytes for inspection."""
        self.push_mode = "nonzero"
        with self.assertRaisesRegex(RuntimeError, "DVC command failed.*9"):
            candidate.Phase("registration", self.work).run(self.context)
        result = self.phase_result(self.work)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["remote_transfer_outcome"], "unknown")
        self.assertEqual(result["dvc_pull_attempts"], 0)
        record = json.loads((self.work / ("command_%02d.json" % result["command_count"])).read_text())
        self.assertEqual((record["exit_code"], record["stdout"], record["stderr"]), (9, "fixture stdout", "fixture push error"))
        self.assertEqual(self.remote_object.read_bytes(), b"partial remote bytes")
        self.assertEqual(self.source.read_bytes(), self.content)
        self.assertTrue((self.work / candidate.POINTER).is_file())
        self.assertTrue((self.work / ".dvc").is_dir())
        self.assertFalse((self.work / candidate.MANIFEST).exists())
        self.assertEqual(sum(args[0] == "push" for _, args in self.actions), 1)

    def test_successful_push_with_wrong_bytes_keeps_transfer_uncertain(self):
        """Exit zero cannot confirm a transfer whose stored bytes fail identity verification."""
        self.push_mode = "corrupt"
        with self.assertRaisesRegex(RuntimeError, "Candidate byte identity differs"):
            candidate.Phase("registration", self.work).run(self.context)
        self.assertEqual(self.phase_result(self.work)["remote_transfer_outcome"], "unknown")
        self.assertEqual(self.remote_object.read_bytes(), b"partial remote bytes")
        self.assertFalse((self.work / candidate.MANIFEST).exists())
        self.assertEqual(sum(args[0] == "push" for _, args in self.actions), 1)

    def test_pull_timeout_records_exception_and_preserves_partial_recovery(self):
        """An interrupted pull records its exception and is attempted exactly once."""
        work = self.prepare_recovery()
        pointer_before = (work / candidate.POINTER).read_bytes()
        self.pull_mode = "timeout"
        with self.assertRaises(candidate.subprocess.TimeoutExpired):
            candidate.Phase("recovery", work).run(self.context)
        result = self.phase_result(work)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["dvc_pull_attempts"], 1)
        record = json.loads((work / ("command_%02d.json" % result["command_count"])).read_text())
        self.assertIsNone(record["exit_code"])
        self.assertIn("timed out", record["error"])
        self.assertEqual((work / candidate.MODEL).read_bytes(), b"partial recovery")
        self.assertEqual((work / candidate.POINTER).read_bytes(), pointer_before)
        self.assertEqual(self.remote_object.read_bytes(), self.content)
        self.assertEqual(sum(args[0] == "pull" for _, args in self.actions), 1)

    def test_recovered_identity_mismatch_preserves_partial_bytes_without_retry(self):
        """A completed pull with corrupt content fails verification and retains its outputs."""
        work = self.prepare_recovery()
        self.pull_mode = "corrupt"
        with self.assertRaisesRegex(RuntimeError, "Candidate byte identity differs"):
            candidate.Phase("recovery", work).run(self.context)
        self.assertEqual(self.phase_result(work)["status"], "failed")
        self.assertEqual((work / candidate.MODEL).read_bytes(), b"partial recovery")
        self.assertEqual(self.remote_object.read_bytes(), self.content)
        self.assertEqual(sum(args[0] == "pull" for _, args in self.actions), 1)

    def test_main_uses_only_redirected_container_paths(self):
        """The entry point reads temporary context and workspace replacements exclusively."""
        with patch.object(sys, "argv", ["register_candidate", "registration"]), patch("builtins.print"):
            self.assertEqual(candidate.main(), 0)
        result = self.phase_result(self.work)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.source.read_bytes(), self.content)
        self.assertEqual(self.remote_object.read_bytes(), self.content)


if __name__ == "__main__":
    unittest.main()
