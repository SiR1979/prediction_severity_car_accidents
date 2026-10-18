"""Exercise the retained manual importer with temporary files and simulated HTTP."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import requests

from src.data import check_structure


BUCKET_URL = "https://example.invalid/raw/"


@pytest.fixture
def importer(monkeypatch):
    """Load unchanged source with its real sibling helpers and guarded boundaries."""
    prompt = Mock(side_effect=AssertionError("Unexpected input prompt"))
    get = Mock(side_effect=AssertionError("Unexpected HTTP request"))
    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(side_effect=AssertionError("Live HTTP is forbidden in this fixture")),
    )
    # The script imports its sibling by a top-level name; restore that alias after each case.
    monkeypatch.setitem(sys.modules, "check_structure", check_structure)
    specification = importlib.util.spec_from_file_location(
        "_isolated_manual_import", Path(__file__).with_name("import_raw_data.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return SimpleNamespace(module=module, prompt=prompt, get=get)


def test_approved_missing_folder_is_created_before_writing_the_download(tmp_path, importer):
    """The actual folder decision permits parent creation and a new UTF-8 file."""
    folder = tmp_path / "new_parent" / "raw"
    importer.prompt.side_effect = ["Y"]
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=200, text="name;value\nété;1\n")

    result = importer.module.import_raw_data(folder, ["sample.csv"], BUCKET_URL)

    assert result is None
    assert folder.is_dir()
    assert (folder / "sample.csv").read_bytes() == b"name;value\n\xc3\xa9t\xc3\xa9;1\n"
    importer.prompt.assert_called_once_with(
        "raw doesn't exists. Do you want to create it? (y/n): "
    )
    importer.get.assert_called_once_with("https://example.invalid/raw/sample.csv")


def test_existing_folder_receives_exact_utf8_bytes_without_prompting(tmp_path, importer):
    """The importer encodes response text as UTF-8 without altering its newlines."""
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=200, text="ville;名\r\nZürich;東京\n")

    importer.module.import_raw_data(tmp_path, ["unicode.csv"], BUCKET_URL)

    assert (tmp_path / "unicode.csv").read_bytes() == (
        b"ville;\xe5\x90\x8d\r\nZ\xc3\xbcrich;\xe6\x9d\xb1\xe4\xba\xac\n"
    )
    importer.prompt.assert_not_called()
    importer.get.assert_called_once_with("https://example.invalid/raw/unicode.csv")


def test_multiple_files_keep_request_order_and_distinct_contents(tmp_path, importer):
    """Two controlled responses produce only the two corresponding output files."""
    importer.get.side_effect = [
        SimpleNamespace(status_code=200, text="first;1\n"),
        SimpleNamespace(status_code=200, text="second;2\n"),
    ]

    importer.module.import_raw_data(tmp_path, ["first.csv", "second.csv"], BUCKET_URL)

    assert importer.get.call_args_list == [
        call("https://example.invalid/raw/first.csv"),
        call("https://example.invalid/raw/second.csv"),
    ]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "first.csv": b"first;1\n",
        "second.csv": b"second;2\n",
    }
    importer.prompt.assert_not_called()


@pytest.mark.parametrize("status", [403, 404, 503])
def test_http_failure_is_reported_without_overwriting_existing_bytes(
    tmp_path, importer, capsys, status
):
    """An approved overwrite still preserves the existing file when HTTP fails."""
    target = tmp_path / "existing.csv"
    original = b"preserved;\x00\xff\r\n"
    target.write_bytes(original)
    importer.prompt.side_effect = ["y"]
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=status, text="do not write")

    result = importer.module.import_raw_data(tmp_path, ["existing.csv"], BUCKET_URL)

    assert result is None
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
    assert f"Error accessing the object https://example.invalid/raw/existing.csv: {status}" in (
        capsys.readouterr().out
    )
    importer.prompt.assert_called_once()
    importer.get.assert_called_once_with("https://example.invalid/raw/existing.csv")


def test_http_failure_does_not_create_a_missing_output_file(tmp_path, importer, capsys):
    """A non-success response reports its status and leaves the directory empty."""
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=404, text="not a CSV")

    importer.module.import_raw_data(tmp_path, ["missing.csv"], BUCKET_URL)

    assert list(tmp_path.iterdir()) == []
    assert "Error accessing the object https://example.invalid/raw/missing.csv: 404" in (
        capsys.readouterr().out
    )
    importer.prompt.assert_not_called()
    importer.get.assert_called_once_with("https://example.invalid/raw/missing.csv")


@pytest.mark.parametrize("answer", ["n", "N"])
def test_declined_overwrite_skips_http_and_preserves_the_file(tmp_path, importer, answer):
    """A declined replacement returns without requesting or changing existing data."""
    target = tmp_path / "existing.csv"
    original = b"keep existing bytes\x00\r\n"
    target.write_bytes(original)
    importer.prompt.side_effect = [answer]

    importer.module.import_raw_data(tmp_path, ["existing.csv"], BUCKET_URL)

    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
    importer.prompt.assert_called_once_with(
        "File existing.csv already exists. Do you want to overwrite it? (y/n): "
    )
    importer.get.assert_not_called()


@pytest.mark.parametrize("answer", ["y", "Y"])
def test_approved_overwrite_replaces_existing_contents(tmp_path, importer, answer):
    """Approval replaces the complete old content, including any longer old suffix."""
    target = tmp_path / "existing.csv"
    target.write_bytes(b"old content with a longer suffix\r\n")
    importer.prompt.side_effect = [answer]
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=200, text="new;é\n")

    importer.module.import_raw_data(tmp_path, ["existing.csv"], BUCKET_URL)

    assert target.read_bytes() == b"new;\xc3\xa9\n"
    assert list(tmp_path.iterdir()) == [target]
    importer.prompt.assert_called_once()
    importer.get.assert_called_once_with("https://example.invalid/raw/existing.csv")


def test_connection_error_propagates_without_creating_or_retrying_output(tmp_path, importer):
    """A simulated connection error escapes unchanged before an output is opened."""
    error = requests.exceptions.ConnectionError("simulated connection failure")
    importer.get.side_effect = error

    with pytest.raises(requests.exceptions.ConnectionError) as caught:
        importer.module.import_raw_data(tmp_path, ["sample.csv"], BUCKET_URL)

    assert caught.value is error
    assert list(tmp_path.iterdir()) == []
    importer.prompt.assert_not_called()
    importer.get.assert_called_once_with("https://example.invalid/raw/sample.csv")


@pytest.mark.parametrize("folder_exists", [True, False])
def test_empty_file_list_performs_no_download(tmp_path, importer, folder_exists):
    """An empty list does no HTTP work, though approved folder creation still occurs."""
    folder = tmp_path / "raw"
    if folder_exists:
        folder.mkdir()
    else:
        importer.prompt.side_effect = ["y"]

    result = importer.module.import_raw_data(folder, [], BUCKET_URL)

    assert result is None
    assert folder.is_dir()
    assert list(folder.iterdir()) == []
    importer.get.assert_not_called()
    if folder_exists:
        importer.prompt.assert_not_called()
    else:
        importer.prompt.assert_called_once()


def test_declined_folder_creation_still_attempts_download_then_fails(tmp_path, importer):
    """Characterize the inherited continuation after refusal without endorsing it."""
    folder = tmp_path / "missing"
    importer.prompt.side_effect = ["n"]
    importer.get.side_effect = None
    importer.get.return_value = SimpleNamespace(status_code=200, text="sample;1\n")

    with pytest.raises(FileNotFoundError):
        importer.module.import_raw_data(folder, ["sample.csv"], BUCKET_URL)

    assert not folder.exists()
    assert list(tmp_path.iterdir()) == []
    importer.prompt.assert_called_once()
    importer.get.assert_called_once_with("https://example.invalid/raw/sample.csv")
