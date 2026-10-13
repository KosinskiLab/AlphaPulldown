"""The expensive GPU tests run only when the code they cover has changed."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "code_changes", Path(__file__).resolve().parents[2] / "test" / "cluster" / "code_changes.py"
)
code_changes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(code_changes)

COVERED = ("alphapulldown/features", "alphafold")


@pytest.fixture
def changed(monkeypatch):
    monkeypatch.delenv(code_changes.FORCE, raising=False)

    def set_changed(paths):
        monkeypatch.setattr(code_changes, "changed_paths", lambda base="origin/main": paths)

    return set_changed


def test_runs_when_a_covered_file_or_submodule_changed(changed):
    changed({"alphapulldown/features/feature_batch.py"})
    assert code_changes.skip_unless_changed(COVERED) is None
    changed({"alphafold"})  # a submodule pointer
    assert code_changes.skip_unless_changed(COVERED) is None


def test_skips_when_only_other_code_changed(changed):
    # "alphafold3" shares a prefix with the covered "alphafold" but is another directory.
    changed({"alphapulldown/folding_backend/alphafold3_backend.py", "alphafold3/setup.py"})
    reason = code_changes.skip_unless_changed(COVERED)
    assert reason and code_changes.FORCE in reason


def test_runs_when_forced_or_when_git_cannot_tell(changed, monkeypatch):
    changed(set())
    monkeypatch.setenv(code_changes.FORCE, "1")
    assert code_changes.skip_unless_changed(COVERED) is None

    monkeypatch.delenv(code_changes.FORCE)
    changed(None)
    assert code_changes.skip_unless_changed(COVERED) is None


def test_changed_paths_is_unknown_outside_a_git_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(code_changes, "REPO_ROOT", tmp_path)
    assert code_changes.changed_paths() is None


def test_covered_paths_exist():
    root = code_changes.REPO_ROOT
    for path in code_changes.FEATURE_GENERATION + code_changes.FAST_KERNELS:
        assert (root / path).exists(), path
