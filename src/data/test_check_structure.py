"""Characterize manual import decisions without changing files or creating paths."""

from unittest.mock import Mock

import pytest

from src.data import check_structure


@pytest.fixture(autouse=True)
def prompt(monkeypatch):
    """Reject unexpected prompts; each interactive case supplies bounded answers."""
    controlled = Mock(side_effect=AssertionError("Unexpected input prompt"))
    monkeypatch.setattr("builtins.input", controlled)
    return controlled


def test_missing_file_is_approved_without_prompt_or_path_creation(tmp_path, prompt):
    """A missing file is approved without creating either it or its missing parent."""
    parent = tmp_path / "missing_parent"
    target = parent / "sample.csv"

    assert check_structure.check_existing_file(target) is True

    prompt.assert_not_called()
    assert not parent.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("answer, expected", [("y", True), ("Y", True), ("n", False), ("N", False)])
def test_existing_file_decision_is_case_insensitive_and_preserves_bytes(
    tmp_path, prompt, answer, expected
):
    """Single-letter decisions return the expected boolean without editing the file."""
    target = tmp_path / "sample.csv"
    original = b"original;\xc3\xa9\r\n\x00"
    target.write_bytes(original)
    prompt.side_effect = None
    prompt.return_value = answer

    assert check_structure.check_existing_file(target) is expected

    prompt.assert_called_once_with(
        "File sample.csv already exists. Do you want to overwrite it? (y/n): "
    )
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("answer, expected", [("Y", True), ("N", False)])
def test_existing_file_retries_invalid_answers_without_changing_content(
    tmp_path, prompt, capsys, answer, expected
):
    """Words and surrounding spaces remain invalid until a supported answer arrives."""
    target = tmp_path / "sample.csv"
    original = b"preserved input\r\n"
    target.write_bytes(original)
    prompt.side_effect = ["yes", " y ", answer]

    assert check_structure.check_existing_file(target) is expected

    assert prompt.call_count == 3
    assert capsys.readouterr().out.splitlines() == [
        "Invalid response. Please enter 'y' or 'n'.",
        "Invalid response. Please enter 'y' or 'n'.",
    ]
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_existing_folder_requires_no_creation_decision_and_preserves_contents(tmp_path, prompt):
    """An existing directory returns False and retains its existing child bytes."""
    folder = tmp_path / "existing"
    folder.mkdir()
    sentinel = folder / "sentinel.bin"
    original = b"existing folder content\x00"
    sentinel.write_bytes(original)

    assert check_structure.check_existing_folder(folder) is False

    prompt.assert_not_called()
    assert list(folder.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == original


@pytest.mark.parametrize("answer, expected", [("y", True), ("Y", True), ("n", False), ("N", False)])
def test_missing_folder_decision_is_case_insensitive_and_does_not_create_paths(
    tmp_path, prompt, answer, expected
):
    """Approval is a decision only; neither answer creates a missing directory."""
    parent = tmp_path / "missing_parent"
    folder = parent / "raw"
    prompt.side_effect = None
    prompt.return_value = answer

    assert check_structure.check_existing_folder(folder) is expected

    prompt.assert_called_once_with("raw doesn't exists. Do you want to create it? (y/n): ")
    assert not parent.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("answer, expected", [("Y", True), ("N", False)])
def test_missing_folder_retries_invalid_answers_without_creating_it(
    tmp_path, prompt, capsys, answer, expected
):
    """Invalid answers cause another prompt and never create the requested path."""
    folder = tmp_path / "raw"
    prompt.side_effect = ["", "no", answer]

    assert check_structure.check_existing_folder(folder) is expected

    assert prompt.call_count == 3
    assert capsys.readouterr().out.splitlines() == [
        "Invalid response. Please enter 'y' or 'n'.",
        "Invalid response. Please enter 'y' or 'n'.",
    ]
    assert not folder.exists()
    assert list(tmp_path.iterdir()) == []
