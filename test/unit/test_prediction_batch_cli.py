import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown/scripts/run_structure_prediction_batch.py"
)


def _load_batch_command(monkeypatch):
    from alphapulldown import scripts

    prediction_command = ModuleType(
        "alphapulldown.scripts.run_structure_prediction"
    )
    prediction_command.backend = object()
    prediction_command._validate_flags_for_backend = lambda _backend: None
    monkeypatch.setitem(
        sys.modules,
        "alphapulldown.scripts.run_structure_prediction",
        prediction_command,
    )
    monkeypatch.setattr(
        scripts, "run_structure_prediction", prediction_command, raising=False
    )

    spec = importlib.util.spec_from_file_location(
        "test_prediction_batch_command", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_reports_rejected_manifest_and_exits_without_a_traceback(
    monkeypatch, tmp_path
):
    command = _load_batch_command(monkeypatch)
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text("{invalid}\n", encoding="utf-8")
    command.FLAGS = SimpleNamespace(
        fold_backend="alphafold2", manifest=str(manifest)
    )
    messages = []
    monkeypatch.setattr(
        command.logging,
        "error",
        lambda template, *args: messages.append(template % args),
    )
    monkeypatch.setattr(
        command.logging,
        "info",
        lambda template, *args: messages.append(template % args),
    )

    with pytest.raises(SystemExit) as exit_info:
        command.main([])

    assert exit_info.value.code == 2
    assert messages == [
        "Prediction batch rejected: Invalid JSON on manifest line 1: "
        "Expecting property name enclosed in double quotes",
        "Prediction batch summary: 0 completed, batch rejected",
    ]
