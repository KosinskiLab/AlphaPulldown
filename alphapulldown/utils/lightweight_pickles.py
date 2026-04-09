"""Helpers for reading AlphaPulldown pickles without importing heavy runtime modules."""

from __future__ import annotations

import gzip
import lzma
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LightweightMonomericObject:
    """Pickle-compatible stand-in for alphapulldown.objects.MonomericObject."""

    description: str = ""
    sequence: str = ""
    feature_dict: dict[str, Any] = field(default_factory=dict)
    _uniprot_runner: Any = None


@dataclass
class LightweightChoppedObject(LightweightMonomericObject):
    """Pickle-compatible stand-in for alphapulldown.objects.ChoppedObject."""

    monomeric_description: str | None = None
    regions: Any = None


class _AlphaPulldownObjectUnpickler(pickle.Unpickler):
    """Unpickler that swaps heavy AlphaPulldown classes for lightweight stand-ins."""

    _CLASS_MAP = {
        ("alphapulldown.objects", "MonomericObject"): LightweightMonomericObject,
        ("alphapulldown.objects", "ChoppedObject"): LightweightChoppedObject,
    }

    def find_class(self, module: str, name: str) -> Any:
        replacement = self._CLASS_MAP.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


def load_lightweight_pickle(path: str | Path) -> Any:
    """Loads a pickle while avoiding imports from alphapulldown.objects."""

    pickle_path = Path(path)
    if pickle_path.suffix == ".xz":
        opener = lzma.open
    elif pickle_path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(pickle_path, "rb") as handle:
        return _AlphaPulldownObjectUnpickler(handle).load()


def extract_feature_dict(payload: Any) -> dict[str, Any]:
    """Returns a feature dictionary from either a raw dict or a monomer-like object."""

    if isinstance(payload, dict):
        return payload

    feature_dict = getattr(payload, "feature_dict", None)
    if not isinstance(feature_dict, dict):
        raise TypeError(
            f"Expected a dict-like payload or an object with feature_dict, got {type(payload)!r}"
        )
    return feature_dict
