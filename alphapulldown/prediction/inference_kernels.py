"""Which kernels predicted a fold: ``alphapulldown/inference_kernels.json`` in its directory.

Fused and original kernels differ slightly in their numerics, so the record lets ranking
and analysis tell predictions apart. Each backend adds one entry per unit it predicts,
keyed ``model_<name>`` by AlphaFold 2 and ``seed-<N>`` by AlphaFold 3. Versions up to
2.10 kept the file at the top of the fold's directory; the next entry written moves it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

from alphapulldown.utils.file_handling import open_atomic
from alphapulldown.utils.output_paths import alphapulldown_dir

INFERENCE_KERNELS = "inference_kernels.json"


def inference_kernels_path(output_dir) -> str:
    """Where the record of the fold in ``output_dir`` is kept."""
    return os.path.join(alphapulldown_dir(output_dir), INFERENCE_KERNELS)


def _legacy_path(output_dir) -> str:
    return os.path.join(os.fspath(output_dir), INFERENCE_KERNELS)


def read_inference_kernels(output_dir) -> dict[str, Any]:
    """The fold's record, wherever this version or an earlier one kept it; empty if
    there is none or it is unreadable."""
    for path in (inference_kernels_path(output_dir), _legacy_path(output_dir)):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            logging.warning("Replacing unreadable %s: %s", path, exc)
            return {}
        return records
    return {}


def record_inference_kernels(output_dir, key: str, entry: Mapping[str, Any]) -> None:
    """Add ``entry`` under ``key``, keeping the other entries. Provenance never fails a
    finished prediction: an unreadable record is replaced, a failed write only logged."""
    records = read_inference_kernels(output_dir)
    records[key] = dict(entry)
    try:
        os.makedirs(alphapulldown_dir(output_dir), exist_ok=True)
        with open_atomic(inference_kernels_path(output_dir)) as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if os.path.exists(_legacy_path(output_dir)):
            os.remove(_legacy_path(output_dir))
    except OSError as exc:
        logging.warning("Not recording the inference kernels of %s: %s", key, exc)
