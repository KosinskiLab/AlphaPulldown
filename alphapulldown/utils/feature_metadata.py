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
from typing import Any, Mapping, Sequence

from alphapulldown.utils.output_paths import FEATURE_METADATA_DIR


AF3_METADATA_MARKER = "\n__ALPHAPULLDOWN_FEATURE_METADATA_V1__="
_AF3_POLYMER_TYPES = ("protein", "rna", "dna")


def retain_msa_provenance(metadata, previous):
    """Combine new template provenance with the retained MSA's original source.

    Callers must collect *template-only* current metadata. Missing historical
    provenance remains unknown; configured current MSA databases are not a
    substitute. Keep this record flat across repeated template refreshes.
    """
    result = copy.deepcopy(metadata)
    previous = previous or {}
    old = previous.get("msa_provenance", {})
    flags = previous.get("other", {})
    msa_names = {"UniRef90", "UniProt", "MGnify", "BFD", "Reduced BFD", "UniRef30",
                 "ColabFold", "NT-RNA", "Rfam", "RNAcentral"}
    inherited = {name: copy.deepcopy(entry) for name, entry in previous.get("databases", {}).items()
                 if name in msa_names}
    if "ColabFold" not in old.get("inherited_databases", ()) and str(flags.get("use_mmseqs2")).lower() != "true":
        inherited.pop("ColabFold", None)
    # Older AF2 writers recorded unused default paths too. AF3 uses Reduced BFD
    # regardless of the legacy AF2 preset flag, so apply this only to AF2 sources.
    if not old and flags.get("data_pipeline", "alphafold2") == "alphafold2":
        if flags.get("db_preset") == "full_dbs":
            inherited.pop("Reduced BFD", None)
        elif flags.get("db_preset") == "reduced_dbs":
            inherited.pop("BFD", None)
            inherited.pop("UniRef30", None)
    result.setdefault("databases", {}).update(inherited)
    result["msa_provenance"] = {
        "mode": "reused_features",
        "source_date": old.get("source_date", previous.get("date")),
        "source_software": copy.deepcopy(old.get("source_software", previous.get("software", {}))),
        "inherited_databases": sorted(inherited),
        "cached_databases": old.get("cached_databases", []),
        "unknown_source": not bool(inherited) or old.get("unknown_source", False),
    }
    # Local MMseqs2 batches describe their searched database snapshots in
    # other.mmseqs2_gpu rather than the legacy databases mapping.
    backend = old.get("source_backend")
    if backend is None and flags.get("mmseqs2_gpu"):
        backend = {key: copy.deepcopy(flags[key]) for key in ("msa_backend", "mmseqs2_gpu") if key in flags}
    if backend is not None:
        result["msa_provenance"]["source_backend"] = copy.deepcopy(backend)
        result["msa_provenance"]["unknown_source"] = old.get("unknown_source", False)
    return result


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


def copied_feature_metadata_paths(directory: str | Path) -> list[Path]:
    """The AF2 metadata sidecars copied into a prediction directory: in
    ``alphapulldown/feature_metadata/``, or at its top where versions up to 2.10 put them."""
    root = Path(directory)
    paths: set[Path] = set()
    for location in (root / FEATURE_METADATA_DIR, root):
        paths |= set(location.glob("*_feature_metadata_*.json"))
        paths |= set(location.glob("*_feature_metadata_*.json.xz"))
    return sorted(paths)


def load_feature_metadata_sidecars(
    directory: str | Path,
) -> list[dict[str, Any]]:
    """Load all AF2 metadata sidecars copied into a prediction directory."""
    return [read_feature_metadata(path) for path in copied_feature_metadata_paths(directory)]
