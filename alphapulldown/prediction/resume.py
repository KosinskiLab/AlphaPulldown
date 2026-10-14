"""Writing prediction outputs so that an interrupted run can be resumed.

Each fold directory holds a record of the run that predicted it
(``alphapulldown/prediction.json``): the settings that shape the fold's outputs, and
whether the run finished. A backend starts the record before predicting and marks it
complete as the last step of post-processing, so a later run with ``--allow_resume`` can
skip a finished fold without inspecting, or redoing, any of its outputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Union

from absl import logging

from alphapulldown.utils.file_handling import open_atomic
from alphapulldown.utils.output_paths import alphapulldown_dir

PathLike = Union[str, "os.PathLike[str]"]

FOLD_RECORD_NAME = "prediction.json"


def fold_record_path(output_dir: PathLike) -> str:
    """Where the record of the fold in ``output_dir`` is kept."""
    return os.path.join(alphapulldown_dir(output_dir), FOLD_RECORD_NAME)


def _as_recorded(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """``settings`` as they read back from the record: tuples become lists, enums text."""
    return json.loads(json.dumps(dict(settings), default=str, sort_keys=True))


def read_fold_record(output_dir: PathLike) -> Dict[str, Any]:
    """The record of the run that predicted the fold in ``output_dir``; empty if none."""
    path = fold_record_path(output_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logging.warning(f"Ignoring unreadable {path}: {exc}")
        return {}
    return record if isinstance(record, dict) else {}


def _write_fold_record(output_dir: PathLike, record: Mapping[str, Any]) -> None:
    from alphapulldown import __version__

    os.makedirs(alphapulldown_dir(output_dir), exist_ok=True)
    with open_atomic(fold_record_path(output_dir)) as handle:
        json.dump(
            {**record, "alphapulldown_version": __version__}, handle, indent=2, sort_keys=True
        )
        handle.write("\n")


def start_fold_record(output_dir: PathLike, settings: Mapping[str, Any]) -> None:
    """Record that the fold in ``output_dir`` is being predicted with ``settings``."""
    _write_fold_record(output_dir, {"settings": _as_recorded(settings), "complete": False})


def finish_fold_record(output_dir: PathLike) -> None:
    """Mark the fold started by ``start_fold_record`` complete; call after its last output.

    A fold predicted without a record (post-processing called on its own) stays without.
    """
    record = read_fold_record(output_dir)
    if record:
        _write_fold_record(output_dir, {**record, "complete": True})


def record_fold_complete(output_dir: PathLike, settings: Mapping[str, Any]) -> None:
    """Record a fold found complete without a record: output of an older version."""
    _write_fold_record(output_dir, {"settings": _as_recorded(settings), "complete": True})


def changed_settings(record: Mapping[str, Any], settings: Mapping[str, Any]) -> List[str]:
    """The names of the settings that differ from those in ``record``."""
    recorded = record.get("settings") or {}
    current = _as_recorded(settings)
    return sorted(
        name for name in set(recorded) | set(current)
        if recorded.get(name) != current.get(name)
    )


def finished_fold(
    record: Mapping[str, Any], settings: Mapping[str, Any], name: str, output_dir: PathLike
) -> bool:
    """Whether ``record`` is of a run that finished this fold with these ``settings``;
    logs which settings changed when they differ."""
    if not record.get("complete"):
        return False
    changes = changed_settings(record, settings)
    if changes:
        logging.info(f"{name} in {output_dir} was finished with other settings: {', '.join(changes)}.")
        return False
    logging.info(f"{name} is already complete in {output_dir}.")
    return True
