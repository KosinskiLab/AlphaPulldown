"""Transport feature-generation metadata across AlphaFold input formats.

AlphaFold 2 feature metadata is stored in sidecar JSON files.  AlphaFold 3's
input schema rejects unknown JSON keys, but permits a free-text ``description``
on polymer entities.  A compact, versioned envelope in that field keeps files
valid for unmodified AlphaFold 3 while allowing AlphaPulldown to recover the
metadata during post-processing.
"""

from __future__ import annotations

import copy
import json
import lzma
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AF3_METADATA_MARKER = "\n__ALPHAPULLDOWN_FEATURE_METADATA_V1__="
_AF3_POLYMER_TYPES = ("protein", "rna", "dna")


def encode_metadata_in_description(
    description: str | None,
    metadata: Mapping[str, Any],
) -> str:
    """Return an AF3-compatible description containing ``metadata``.

    An existing AlphaPulldown envelope is replaced, making repeated embedding
    idempotent.  The original user-facing description is retained verbatim.
    """
    clean_description, _ = decode_metadata_from_description(description)
    envelope = {
        "schema_version": 1,
        "metadata": dict(metadata),
    }
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{clean_description or ''}{AF3_METADATA_MARKER}{encoded}"


def decode_metadata_from_description(
    description: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Split a polymer description into its original text and AP metadata."""
    if not isinstance(description, str) or AF3_METADATA_MARKER not in description:
        return description, None

    clean_description, encoded = description.rsplit(AF3_METADATA_MARKER, 1)
    try:
        envelope = json.loads(encoded)
    except (TypeError, ValueError):
        return description, None

    if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
        return description, None
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        return description, None
    return clean_description or None, metadata


def embed_metadata_in_af3_json(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Embed metadata in the first polymer description of an AF3 JSON job.

    Only keys already accepted by vanilla AlphaFold 3 are used.  A copy is
    returned so callers never mutate an object supplied by another library.

    Raises:
        ValueError: If the payload has no AF3 polymer entity in ``sequences``.
    """
    output = copy.deepcopy(dict(payload))
    sequences = output.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("AF3 feature JSON does not contain a sequences list")

    for sequence_entry in sequences:
        if not isinstance(sequence_entry, dict):
            continue
        for polymer_type in _AF3_POLYMER_TYPES:
            polymer = sequence_entry.get(polymer_type)
            if not isinstance(polymer, dict):
                continue
            polymer["description"] = encode_metadata_in_description(
                polymer.get("description"), metadata
            )
            return output

    raise ValueError("AF3 feature JSON does not contain a polymer entity")


def extract_metadata_from_af3_json(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return all AlphaPulldown metadata envelopes found in an AF3 JSON job."""
    found: list[dict[str, Any]] = []
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        return found

    for sequence_entry in sequences:
        if not isinstance(sequence_entry, dict):
            continue
        for polymer_type in _AF3_POLYMER_TYPES:
            polymer = sequence_entry.get(polymer_type)
            if not isinstance(polymer, dict):
                continue
            _, metadata = decode_metadata_from_description(
                polymer.get("description")
            )
            if metadata is not None:
                found.append(metadata)
    return found


def extract_metadata_from_fold_input(fold_input: Any) -> list[dict[str, Any]]:
    """Recover embedded metadata from an AF3 ``folding_input.Input`` object."""
    found: list[dict[str, Any]] = []
    for chain in getattr(fold_input, "chains", ()):
        _, metadata = decode_metadata_from_description(
            getattr(chain, "description", None)
        )
        if metadata is not None:
            found.append(metadata)
    return found


def read_feature_metadata(path: str | Path) -> dict[str, Any]:
    """Read an AF2 feature metadata sidecar, compressed or uncompressed."""
    metadata_path = Path(path)
    opener = lzma.open if metadata_path.suffix == ".xz" else open
    with opener(metadata_path, "rt", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Feature metadata in {metadata_path} is not a JSON object")
    return metadata


def _normalise_feature_directories(
    feature_directories: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if feature_directories is None:
        return []
    if isinstance(feature_directories, (str, Path)):
        return [Path(feature_directories)]
    return [Path(path) for path in feature_directories]


def find_feature_metadata(
    description: str,
    feature_directories: str | Path | Sequence[str | Path] | None,
) -> dict[str, Any] | None:
    """Load the newest AF2 metadata sidecar for ``description`` if present."""
    matches: list[Path] = []
    for feature_dir in _normalise_feature_directories(feature_directories):
        matches.extend(feature_dir.glob(f"{description}_feature_metadata_*.json"))
        matches.extend(feature_dir.glob(f"{description}_feature_metadata_*.json.xz"))
    if not matches:
        return None
    newest = max(matches, key=lambda path: path.stat().st_mtime)
    return read_feature_metadata(newest)


def load_feature_metadata_sidecars(
    directory: str | Path,
) -> list[dict[str, Any]]:
    """Load all AF2 metadata sidecars copied into a prediction directory."""
    root = Path(directory)
    paths: Iterable[Path] = sorted(
        set(root.glob("*_feature_metadata_*.json"))
        | set(root.glob("*_feature_metadata_*.json.xz"))
    )
    return [read_feature_metadata(path) for path in paths]
