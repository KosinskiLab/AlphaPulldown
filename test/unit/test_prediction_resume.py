"""Outputs a resumed run relies on appear whole or not at all."""

import os
import pickle

import pytest

from alphapulldown.prediction.resume import open_atomic


def _leftovers(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))


def test_open_atomic_replaces_the_file_with_the_whole_new_content(tmp_path):
    target = tmp_path / "timings.json"
    target.write_text("old", encoding="utf-8")

    with open_atomic(target) as handle:
        handle.write("new")

    assert target.read_text(encoding="utf-8") == "new"
    assert _leftovers(tmp_path) == []


def test_open_atomic_keeps_the_previous_file_when_writing_fails(tmp_path):
    target = tmp_path / "unrelaxed_model.pdb"
    target.write_text("complete model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="killed"):
        with open_atomic(target) as handle:
            handle.write("half a mod")
            raise RuntimeError("killed")

    assert target.read_text(encoding="utf-8") == "complete model"
    assert _leftovers(tmp_path) == []


def test_open_atomic_leaves_nothing_when_a_new_file_fails(tmp_path):
    target = tmp_path / "result_model.pkl"

    with pytest.raises(RuntimeError):
        with open_atomic(target, "wb") as handle:
            pickle.dump({"plddt": [1.0]}, handle)
            raise RuntimeError("killed")

    assert list(tmp_path.iterdir()) == []


def test_open_atomic_writes_binary_files(tmp_path):
    target = tmp_path / "result_model.pkl"

    with open_atomic(target, "wb") as handle:
        pickle.dump({"plddt": [1.0, 2.0]}, handle, protocol=4)

    with open(target, "rb") as handle:
        assert pickle.load(handle) == {"plddt": [1.0, 2.0]}


def test_open_atomic_files_follow_the_umask_rather_than_owner_only(tmp_path):
    previous = os.umask(0o022)
    try:
        with open_atomic(tmp_path / "ranked_0.pdb") as handle:
            handle.write("model")
    finally:
        os.umask(previous)

    assert (tmp_path / "ranked_0.pdb").stat().st_mode & 0o777 == 0o644


def test_open_atomic_rejects_partial_writes(tmp_path):
    with pytest.raises(ValueError, match="whole files"):
        with open_atomic(tmp_path / "timings.json", "a"):
            pass
