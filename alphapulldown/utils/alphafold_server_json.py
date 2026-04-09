"""Build AlphaFold Server batch JSON inputs from AlphaPulldown jobs."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from alphapulldown_input_parser import RegionSelection, generate_fold_specifications, parse_fold

from alphapulldown.utils.file_handling import make_dir_monomer_dictionary
from alphapulldown.utils.lightweight_pickles import load_lightweight_pickle


_SERVER_DIALECT = "alphafoldserver"
_SERVER_VERSION = 1
_SERVER_ENTITY_KEYS = ("proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion")


def _json_input_basename(json_input_path: str) -> str:
    stem = Path(json_input_path).stem
    for suffix in ("_af3_input", "_input"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or Path(json_input_path).stem


def _sanitize_job_name(name: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._")
    return sanitized or "alphafold_server_job"


def _regions_to_tuples(selection: RegionSelection | Any) -> str | list[tuple[int, int]]:
    if isinstance(selection, RegionSelection):
        if selection.is_all:
            return "all"
        return [(region.start, region.end) for region in selection.regions]
    return selection


def _normalise_job_entries(job: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for entry in job:
        if "json_input" in entry:
            normalised_entry: dict[str, Any] = {"json_input": entry["json_input"]}
            regions = entry.get("regions")
            if isinstance(regions, RegionSelection) and not regions.is_all:
                normalised_entry["regions"] = _regions_to_tuples(regions)
            normalised.append(normalised_entry)
            continue

        name, selection = next(iter(entry.items()))
        normalised.append({name: _regions_to_tuples(selection)})
    return normalised


def _slice_sequence(sequence: str, regions: list[tuple[int, int]] | None) -> str:
    if not regions:
        return sequence
    chunks = []
    for start, end in regions:
        if start < 1 or end < start:
            raise ValueError(f"Invalid region range {(start, end)}")
        chunks.append(sequence[start - 1 : end])
    return "".join(chunks)


def _resolve_feature_pickle_path(
    monomer_directories: list[str],
    protein_name: str,
) -> Path:
    monomer_dir_map = make_dir_monomer_dictionary(monomer_directories)
    for suffix in (".pkl", ".pkl.xz"):
        filename = f"{protein_name}{suffix}"
        directory = monomer_dir_map.get(filename)
        if directory is not None:
            return Path(directory) / filename
    raise FileNotFoundError(
        f"Could not find a feature pickle for {protein_name!r} in {monomer_directories!r}"
    )


def _protein_entity(sequence: str) -> dict[str, Any]:
    return {"proteinChain": {"sequence": sequence, "count": 1}}


def _sequence_entity(
    entity_key: str,
    sequence: str,
    *,
    count: int = 1,
) -> dict[str, Any]:
    return {entity_key: {"sequence": sequence, "count": count}}


def _simple_entity(entity_key: str, value_key: str, value: str, *, count: int = 1) -> dict[str, Any]:
    return {entity_key: {value_key: value, "count": count}}


def _convert_local_af3_entity(
    entity: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(entity) != 1:
        raise ValueError(f"Expected one entity per AF3 JSON sequence entry, got {entity!r}")

    entity_type, payload = next(iter(entity.items()))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload for {entity_type!r}, got {type(payload)!r}")

    if entity_type == "protein":
        sequence = payload.get("sequence")
        if not isinstance(sequence, str):
            raise ValueError("AF3 protein entities must contain a sequence string")
        return [_protein_entity(sequence)]

    if entity_type == "dna":
        sequence = payload.get("sequence")
        if not isinstance(sequence, str):
            raise ValueError("AF3 DNA entities must contain a sequence string")
        return [_sequence_entity("dnaSequence", sequence)]

    if entity_type == "rna":
        sequence = payload.get("sequence")
        if not isinstance(sequence, str):
            raise ValueError("AF3 RNA entities must contain a sequence string")
        return [_sequence_entity("rnaSequence", sequence)]

    if entity_type == "ligand":
        ccd_codes = payload.get("ccdCodes")
        if not isinstance(ccd_codes, list) or not all(isinstance(code, str) for code in ccd_codes):
            raise ValueError("AF3 ligand entities must provide ccdCodes as a list of strings")
        return [_simple_entity("ligand", "ligand", code) for code in ccd_codes]

    if entity_type == "ion":
        ion_value = payload.get("ion")
        if not isinstance(ion_value, str):
            raise ValueError("AF3 ion entities must contain an ion string")
        return [_simple_entity("ion", "ion", ion_value)]

    raise ValueError(f"Unsupported AF3 entity type {entity_type!r} for AlphaFold Server export")


def _convert_server_entity(entity: dict[str, Any]) -> dict[str, Any]:
    if len(entity) != 1:
        raise ValueError(f"Expected one entity per server JSON sequence entry, got {entity!r}")
    entity_type, payload = next(iter(entity.items()))
    if entity_type not in _SERVER_ENTITY_KEYS:
        raise ValueError(f"Unsupported AlphaFold Server entity type {entity_type!r}")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload for {entity_type!r}, got {type(payload)!r}")
    return {entity_type: deepcopy(payload)}


def _load_json_input_entities(
    json_input_path: str,
) -> list[dict[str, Any]]:
    with open(json_input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"JSON input {json_input_path} contains {len(payload)} jobs; expected exactly one"
            )
        payload = payload[0]

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported JSON root in {json_input_path!r}: {type(payload)!r}")

    dialect = payload.get("dialect")
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError(f"JSON input {json_input_path} does not define a sequences list")

    if dialect == _SERVER_DIALECT:
        return [_convert_server_entity(entity) for entity in sequences]
    if dialect == "alphafold3":
        converted: list[dict[str, Any]] = []
        for entity in sequences:
            if not isinstance(entity, dict):
                raise TypeError(f"Unsupported AF3 entity payload in {json_input_path!r}: {entity!r}")
            converted.extend(_convert_local_af3_entity(entity))
        return converted

    raise ValueError(
        f"Unsupported JSON dialect {dialect!r} in {json_input_path!r}. "
        f"Expected {_SERVER_DIALECT!r} or 'alphafold3'."
    )


def _slice_json_input_entities(
    entities: list[dict[str, Any]],
    json_input_path: str,
    regions: list[tuple[int, int]] | None,
) -> list[dict[str, Any]]:
    if not regions:
        return entities
    if len(entities) != 1:
        raise ValueError(
            "Region ranges for JSON inputs require exactly one entity, but "
            f"{json_input_path!r} contains {len(entities)} entities"
        )
    entity = deepcopy(entities[0])
    entity_type, payload = next(iter(entity.items()))
    if entity_type not in {"proteinChain", "dnaSequence", "rnaSequence"}:
        raise ValueError(
            f"Region slicing is only supported for sequence entities, not {entity_type!r}"
        )
    sequence = payload.get("sequence")
    if not isinstance(sequence, str):
        raise ValueError(f"{entity_type!r} in {json_input_path!r} does not contain a sequence string")
    payload["sequence"] = _slice_sequence(sequence, regions)
    return [entity]


def _collapse_duplicate_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for entity in entities:
        entity_type, payload = next(iter(entity.items()))
        payload_copy = deepcopy(payload)
        count = int(payload_copy.pop("count", 1))
        key = json.dumps({entity_type: payload_copy}, sort_keys=True)
        if key not in collapsed:
            payload_copy["count"] = count
            collapsed[key] = {entity_type: payload_copy}
        else:
            existing_type, existing_payload = next(iter(collapsed[key].items()))
            if existing_type != entity_type:
                raise AssertionError("Entity deduplication key collision across types")
            existing_payload["count"] = int(existing_payload.get("count", 1)) + count
    return list(collapsed.values())


def _build_job_name(name_fragments: list[str]) -> str:
    return _sanitize_job_name("_and_".join(fragment for fragment in name_fragments if fragment))


def build_job_name_from_entries(entries: list[dict[str, Any]]) -> str:
    fragments: list[str] = []
    for entry in entries:
        if "json_input" in entry:
            fragment = _json_input_basename(entry["json_input"])
            regions = entry.get("regions")
            if regions:
                ranges = "_".join(f"{start}-{end}" for start, end in regions)
                fragment = f"{fragment}_{ranges}"
            fragments.append(fragment)
            continue

        protein_name = next(iter(entry))
        fragments.append(protein_name)
    return _build_job_name(fragments)


def build_alphafold_server_job(
    job_entries: list[dict[str, Any]],
    monomer_directories: list[str],
    *,
    model_seeds: list[str] | None = None,
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for entry in job_entries:
        if "json_input" in entry:
            json_entities = _load_json_input_entities(entry["json_input"])
            entities.extend(
                _slice_json_input_entities(
                    json_entities,
                    entry["json_input"],
                    entry.get("regions"),
                )
            )
            continue

        protein_name, selection = next(iter(entry.items()))
        feature_pickle = _resolve_feature_pickle_path(monomer_directories, protein_name)
        payload = load_lightweight_pickle(feature_pickle)
        sequence = getattr(payload, "sequence", None)
        if not isinstance(sequence, str):
            raise ValueError(f"Feature pickle {feature_pickle} does not contain a sequence string")
        regions = None if selection == "all" else selection
        entities.append(_protein_entity(_slice_sequence(sequence, regions)))

    return {
        "name": build_job_name_from_entries(job_entries),
        "modelSeeds": list(model_seeds or []),
        "sequences": _collapse_duplicate_entities(entities),
        "dialect": _SERVER_DIALECT,
        "version": _SERVER_VERSION,
    }


def build_alphafold_server_jobs(
    *,
    protein_lists: list[str],
    monomer_directories: list[str],
    mode: str = "pulldown",
    oligomer_state_file: str | None = None,
    protein_delimiter: str = "+",
    model_seeds: list[str] | None = None,
    job_index: int | None = None,
) -> list[dict[str, Any]]:
    active_lists = list(protein_lists)
    if mode == "all_vs_all":
        active_lists = [protein_lists[0], protein_lists[0]]
    elif mode == "homo-oligomer":
        if oligomer_state_file is None:
            raise ValueError("oligomer_state_file is required for mode='homo-oligomer'")
        active_lists = [oligomer_state_file]

    specifications = generate_fold_specifications(
        input_files=active_lists,
        delimiter=protein_delimiter,
        exclude_permutations=True,
    )
    all_folds = [spec.replace(",", ":").replace(";", "+") for spec in specifications]

    if job_index is not None:
        zero_based_index = job_index - 1
        if zero_based_index < 0 or zero_based_index >= len(all_folds):
            raise IndexError(
                f"job_index must be between 1 and {len(all_folds)}, got {job_index}"
            )
        selected_folds = [all_folds[zero_based_index]]
    else:
        selected_folds = all_folds

    parsed_jobs = parse_fold(selected_folds, monomer_directories, protein_delimiter)
    normalised_jobs = [_normalise_job_entries(job) for job in parsed_jobs]
    return [
        build_alphafold_server_job(
            normalised_job,
            monomer_directories,
            model_seeds=model_seeds,
        )
        for normalised_job in normalised_jobs
    ]


def write_jobs_to_json_files(
    jobs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    jobs_per_file: int = 100,
) -> list[Path]:
    if jobs_per_file < 1:
        raise ValueError("jobs_per_file must be at least 1")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    job_batches = [
        jobs[index : index + jobs_per_file]
        for index in range(0, len(jobs), jobs_per_file)
    ] or [[]]

    written_paths: list[Path] = []
    if len(job_batches) == 1:
        target_paths = [destination]
    else:
        target_paths = [
            destination.with_name(f"{destination.stem}_{index:03d}{destination.suffix or '.json'}")
            for index in range(1, len(job_batches) + 1)
        ]

    for batch, target_path in zip(job_batches, target_paths, strict=True):
        suffix = target_path.suffix or ".json"
        if target_path.suffix != suffix:
            target_path = target_path.with_suffix(suffix)
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(batch, handle, indent=2)
            handle.write("\n")
        written_paths.append(target_path)
    return written_paths

