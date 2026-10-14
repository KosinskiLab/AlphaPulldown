"""Add AlphaPulldown feature provenance to AlphaFold 3 ModelCIF output."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alphapulldown.utils.feature_metadata import decode_metadata_from_description
from alphapulldown.utils.modelcif_parameters import is_modelcif_parameter


_SOFTWARE_DETAILS = {
    "alphapulldown": (
        "model building",
        "AlphaPulldown structure-prediction workflow",
        "package",
        "https://github.com/KosinskiLab/AlphaPulldown",
    ),
    "alphafold 2": (
        "data collection",
        "AlphaFold 2 feature generation for AlphaFold 3 inference",
        "package",
        "https://github.com/google-deepmind/alphafold",
    ),
    "hhblits": (
        "data collection",
        "Iterative protein sequence search",
        "program",
        "https://github.com/soedinglab/hh-suite",
    ),
    "hhsearch": (
        "data collection",
        "Protein template search",
        "program",
        "https://github.com/soedinglab/hh-suite",
    ),
    "jackhmmer": (
        "data collection",
        "Iterative protein sequence search",
        "program",
        "http://hmmer.org/",
    ),
    "nhmmer": (
        "data collection",
        "Nucleotide sequence search",
        "program",
        "http://hmmer.org/",
    ),
    "hmmalign": (
        "data collection",
        "Sequence-to-profile alignment",
        "program",
        "http://hmmer.org/",
    ),
    "hmmsearch": (
        "data collection",
        "Profile sequence search",
        "program",
        "http://hmmer.org/",
    ),
    "hmmbuild": (
        "data collection",
        "Profile HMM construction",
        "program",
        "http://hmmer.org/",
    ),
    "kalign": (
        "data collection",
        "Multiple sequence alignment",
        "program",
        "https://github.com/timolassmann/kalign",
    ),
}


def _as_list(cif: Mapping[str, Sequence[str]], key: str) -> list[str]:
    return [str(value) for value in cif.get(key, ())]


def _metadata_pipeline(metadata: Mapping[str, Any]) -> str:
    other = metadata.get("other", {})
    if isinstance(other, Mapping):
        return str(other.get("data_pipeline", "")).lower()
    return ""


def _software_from_metadata(
    metadata_records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for metadata in metadata_records:
        software = metadata.get("software", {})
        if not isinstance(software, Mapping):
            continue
        pipeline = _metadata_pipeline(metadata)
        for raw_name, details in software.items():
            name = str(raw_name)
            version = "?"
            if isinstance(details, Mapping) and details.get("version") is not None:
                version = str(details["version"])
            if name.lower() == "alphafold":
                if pipeline == "alphafold3":
                    # AF3's own ModelCIF writer already records AlphaFold 3.
                    continue
                if pipeline == "alphafold2" or (
                    not pipeline and version.lstrip("v").startswith("2.")
                ):
                    # Metadata sidecars created before ``data_pipeline`` was a
                    # flag still identify themselves through the AF2 version.
                    name = "AlphaFold 2"
            key = (name.casefold(), version)
            if key not in seen:
                seen.add(key)
                records.append((name, version))
    return records


def _add_software(
    cif: Mapping[str, Sequence[str]],
    updates: dict[str, list[str]],
    metadata_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    columns = {
        "_software.pdbx_ordinal": "?",
        "_software.name": "?",
        "_software.classification": "other",
        "_software.description": ".",
        "_software.version": "?",
        "_software.type": "program",
        "_software.location": "?",
        "_software.date": "?",
    }
    lengths = [len(cif.get(key, ())) for key in columns]
    row_count = max(lengths, default=0)
    table = {}
    for key, default in columns.items():
        values = _as_list(cif, key)
        values.extend([default] * (row_count - len(values)))
        table[key] = values

    existing = {
        (name.casefold(), version)
        for name, version in zip(
            table["_software.name"], table["_software.version"], strict=True
        )
    }
    added_ids: list[str] = []
    alphapulldown_ids: list[str] = []
    for name, version in _software_from_metadata(metadata_records):
        if (name.casefold(), version) in existing:
            continue
        software_id = str(len(table["_software.name"]) + 1)
        classification, description, software_type, location = _SOFTWARE_DETAILS.get(
            name.casefold(),
            ("data collection", "Feature generation", "program", "?"),
        )
        values = {
            "_software.pdbx_ordinal": software_id,
            "_software.name": name,
            "_software.classification": classification,
            "_software.description": description,
            "_software.version": version,
            "_software.type": software_type,
            "_software.location": location,
            "_software.date": "?",
        }
        for key in columns:
            table[key].append(values[key])
        existing.add((name.casefold(), version))
        added_ids.append(software_id)
        if name.casefold() == "alphapulldown":
            alphapulldown_ids.append(software_id)

    updates.update(table)
    return added_ids, alphapulldown_ids


def _parameter_value(value: str) -> tuple[str, str]:
    if value in {"True", "False"}:
        return "boolean", "YES" if value == "True" else "NO"
    if re.fullmatch(r"[+-]?\d+", value):
        return "integer", value
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value):
        return "float", value
    return "string", value


def _add_parameters(
    updates: dict[str, list[str]],
    metadata_records: Sequence[Mapping[str, Any]],
) -> str | None:
    values_by_name: defaultdict[str, set[str]] = defaultdict(set)
    for metadata in metadata_records:
        other = metadata.get("other", {})
        if not isinstance(other, Mapping):
            continue
        for name, value in other.items():
            name = str(name)
            if is_modelcif_parameter(name):
                values_by_name[name].add(str(value))
    if not values_by_name:
        return None

    parameter_ids: list[str] = []
    group_ids: list[str] = []
    data_types: list[str] = []
    names: list[str] = []
    values: list[str] = []
    descriptions: list[str] = []
    for parameter_id, name in enumerate(sorted(values_by_name), start=1):
        raw_values = sorted(values_by_name[name])
        raw_value = raw_values[0] if len(raw_values) == 1 else json.dumps(raw_values)
        data_type, normalised_value = _parameter_value(raw_value)
        parameter_ids.append(str(parameter_id))
        group_ids.append("1")
        data_types.append(data_type)
        names.append(f"--{name}")
        values.append(normalised_value)
        descriptions.append(".")

    updates.update(
        {
            "_ma_software_parameter.parameter_id": parameter_ids,
            "_ma_software_parameter.group_id": group_ids,
            "_ma_software_parameter.data_type": data_types,
            "_ma_software_parameter.name": names,
            "_ma_software_parameter.value": values,
            "_ma_software_parameter.description": descriptions,
        }
    )
    return "1"


def _add_software_groups(
    cif: Mapping[str, Sequence[str]],
    updates: dict[str, list[str]],
    added_software_ids: Sequence[str],
    alphapulldown_ids: Sequence[str],
    parameter_group_id: str | None,
) -> None:
    ordinals = _as_list(cif, "_ma_software_group.ordinal_id")
    group_ids = _as_list(cif, "_ma_software_group.group_id")
    software_ids = _as_list(cif, "_ma_software_group.software_id")
    parameter_ids = _as_list(cif, "_ma_software_group.parameter_group_id")
    parameter_ids.extend(["."] * (len(ordinals) - len(parameter_ids)))

    def append(group_id: str, software_id: str, parameters: str = ".") -> None:
        ordinals.append(str(len(ordinals) + 1))
        group_ids.append(group_id)
        software_ids.append(software_id)
        parameter_ids.append(parameters)

    for software_id in added_software_ids:
        append(
            "2",
            software_id,
            parameter_group_id if software_id in alphapulldown_ids and parameter_group_id else ".",
        )

    if not added_software_ids and software_ids:
        # Metadata with only AlphaFold recorded still needs a valid feature
        # preparation group for the protocol links below.
        append("2", software_ids[0])

    # The modeling group records the AF3 software already in group 1 plus the
    # AlphaPulldown workflow that invoked it.
    model_software_ids = list(dict.fromkeys(software_ids[:1] + list(alphapulldown_ids)))
    for software_id in model_software_ids:
        append("3", software_id)

    updates.update(
        {
            "_ma_software_group.ordinal_id": ordinals,
            "_ma_software_group.group_id": group_ids,
            "_ma_software_group.software_id": software_ids,
            "_ma_software_group.parameter_group_id": parameter_ids,
        }
    )


def _database_rows(
    metadata_records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for metadata in metadata_records:
        databases = metadata.get("databases", {})
        if not isinstance(databases, Mapping):
            continue
        for name, details in databases.items():
            if not isinstance(details, Mapping):
                continue
            urls = details.get("location_url") or ["?"]
            if isinstance(urls, str):
                urls = [urls]
            version = "?" if details.get("version") is None else str(details["version"])
            release_date = (
                "."
                if details.get("release_date") in (None, "NA", "AF2")
                else str(details["release_date"])[:10]
            )
            for url in urls:
                row = (str(name), str(url), version, release_date)
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
    return rows


def _add_databases(
    cif: Mapping[str, Sequence[str]],
    updates: dict[str, list[str]],
    metadata_records: Sequence[Mapping[str, Any]],
) -> None:
    database_rows = _database_rows(metadata_records)
    if not database_rows:
        return

    data_ids = _as_list(cif, "_ma_data.id")
    data_names = _as_list(cif, "_ma_data.name")
    content_types = _as_list(cif, "_ma_data.content_type")
    other_details = _as_list(cif, "_ma_data.content_type_other_details")
    other_details.extend(["."] * (len(data_ids) - len(other_details)))

    ref_ids = _as_list(cif, "_ma_data_ref_db.data_id")
    ref_names = _as_list(cif, "_ma_data_ref_db.name")
    ref_urls = _as_list(cif, "_ma_data_ref_db.location_url")
    ref_versions = _as_list(cif, "_ma_data_ref_db.version")
    ref_dates = _as_list(cif, "_ma_data_ref_db.release_date")
    new_ref_ids: list[str] = []
    for name, url, version, release_date in database_rows:
        data_id = str(len(data_ids) + 1)
        data_ids.append(data_id)
        data_names.append(name)
        content_types.append("reference database")
        other_details.append(".")
        ref_ids.append(data_id)
        new_ref_ids.append(data_id)
        ref_names.append(name)
        ref_urls.append(url)
        ref_versions.append(version)
        ref_dates.append(release_date)

    updates.update(
        {
            "_ma_data.id": data_ids,
            "_ma_data.name": data_names,
            "_ma_data.content_type": content_types,
            "_ma_data.content_type_other_details": other_details,
            "_ma_data_ref_db.data_id": ref_ids,
            "_ma_data_ref_db.name": ref_names,
            "_ma_data_ref_db.location_url": ref_urls,
            "_ma_data_ref_db.version": ref_versions,
            "_ma_data_ref_db.release_date": ref_dates,
        }
    )
    group_ordinals = _as_list(cif, "_ma_data_group.ordinal_id")
    group_ids = _as_list(cif, "_ma_data_group.group_id")
    group_data_ids = _as_list(cif, "_ma_data_group.data_id")
    for data_id in new_ref_ids:
        group_ordinals.append(str(len(group_ordinals) + 1))
        group_ids.append("1")
        group_data_ids.append(data_id)
    updates.update(
        {
            "_ma_data_group.ordinal_id": group_ordinals,
            "_ma_data_group.group_id": group_ids,
            "_ma_data_group.data_id": group_data_ids,
        }
    )


def _add_protocol_links(
    cif: Mapping[str, Sequence[str]], updates: dict[str, list[str]]
) -> None:
    method_types = _as_list(cif, "_ma_protocol_step.method_type")
    count = len(method_types)
    software_groups = []
    input_groups = []
    step_names = []
    details = []
    has_input_group = bool(
        updates.get("_ma_data_group.group_id")
        or cif.get("_ma_data_group.group_id", ())
    )
    for method_type in method_types:
        if method_type in {"coevolution MSA", "template search"}:
            software_groups.append("2")
            input_groups.append("1" if has_input_group else ".")
        else:
            software_groups.append("3")
            input_groups.append(".")
        step_names.append(
            {
                "coevolution MSA": "MSA generation",
                "template search": "Template search",
                "modeling": "Modeling",
            }.get(method_type, ".")
        )
        details.append(".")
    if count:
        updates["_ma_protocol_step.software_group_id"] = software_groups
        updates["_ma_protocol_step.input_data_group_id"] = input_groups
        updates["_ma_protocol_step.output_data_group_id"] = ["."] * count
        updates["_ma_protocol_step.step_name"] = step_names
        updates["_ma_protocol_step.details"] = details


def _clean_entity_descriptions(
    cif: Mapping[str, Sequence[str]], updates: dict[str, list[str]]
) -> None:
    """Remove transport envelopes from final ModelCIF entity descriptions."""
    descriptions = _as_list(cif, "_entity.pdbx_description")
    if not descriptions:
        return

    cleaned_descriptions = []
    changed = False
    for description in descriptions:
        clean_description, metadata = decode_metadata_from_description(description)
        if metadata is None:
            cleaned_descriptions.append(description)
            continue
        cleaned_descriptions.append(clean_description or ".")
        changed = True
    if changed:
        updates["_entity.pdbx_description"] = cleaned_descriptions


def build_alphapulldown_mmcif_updates(
    cif: Mapping[str, Sequence[str]],
    metadata_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Build ModelCIF category updates for feature provenance metadata."""
    updates: dict[str, list[str]] = {}
    added_software_ids, alphapulldown_ids = _add_software(
        cif, updates, metadata_records
    )
    parameter_group_id = _add_parameters(updates, metadata_records)
    _add_software_groups(
        cif,
        updates,
        added_software_ids,
        alphapulldown_ids,
        parameter_group_id,
    )
    _add_databases(cif, updates, metadata_records)
    _add_protocol_links(cif, updates)
    _clean_entity_descriptions(cif, updates)
    return updates


def augment_af3_modelcif_file(
    cif_path: str | Path,
    metadata_records: Sequence[Mapping[str, Any]],
) -> bool:
    """Enrich an AF3 ModelCIF file in place, preserving its legal comments."""
    if not metadata_records:
        return False

    from alphafold3.structure import mmcif

    path = Path(cif_path)
    original = path.read_text(encoding="utf-8")
    data_match = re.search(r"(?m)^data_", original)
    prefix = original[: data_match.start()] if data_match else ""
    parsed = mmcif.from_string(original)
    updates = build_alphapulldown_mmcif_updates(parsed, metadata_records)
    rendered = parsed.copy_and_update(updates).to_string()
    path.write_text(prefix + rendered, encoding="utf-8")
    return True


def find_af3_modelcif_files(output_dir: str | Path, job_name: str) -> list[Path]:
    """Return best-model and per-sample AF3 ModelCIF paths."""
    root = Path(output_dir)
    candidates = [root / f"{job_name}_model.cif"]
    candidates.extend(sorted(root.glob("seed-*_sample-*/model.cif")))
    return [path for path in candidates if path.is_file()]
