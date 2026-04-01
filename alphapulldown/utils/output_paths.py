import json
import os
import string
from pathlib import Path
from typing import Sequence


_AF3_ALLOWED_NAME_CHARS = set(string.ascii_lowercase + string.digits + "_-.")


def _raw_sanitise_af3_job_name(job_name: str) -> str:
    lower_spaceless_name = job_name.lower().replace(" ", "_")
    return "".join(ch for ch in lower_spaceless_name if ch in _AF3_ALLOWED_NAME_CHARS)


def sanitise_af3_job_name(job_name: str) -> str:
    """Match AlphaFold 3's filename sanitisation for job names with safe fallbacks."""
    sanitised = _raw_sanitise_af3_job_name(job_name)
    if sanitised in {".", ".."}:
        return "ranked_0"
    return sanitised or "ranked_0"


def derive_af3_job_name_from_json(json_input_path: str) -> str:
    """Derive the AF3 job name from the current JSON input."""
    fallback_name = sanitise_af3_job_name(Path(json_input_path).stem)

    try:
        with open(json_input_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return fallback_name

    if isinstance(payload, dict):
        raw_name = payload.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            sanitised_name = _raw_sanitise_af3_job_name(raw_name)
            if sanitised_name and sanitised_name not in {".", ".."}:
                return sanitised_name

    return fallback_name


def _json_input_basename(json_input_path: str) -> str:
    stem = Path(json_input_path).stem
    for suffix in ("_af3_input", "_input"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or Path(json_input_path).stem


def _collapse_repeated_name_fragments(fragments: Sequence[str]) -> list[str]:
    if not fragments:
        return []

    collapsed: list[str] = []
    current_fragment = fragments[0]
    current_count = 1

    for fragment in fragments[1:]:
        if fragment == current_fragment:
            current_count += 1
            continue

        collapsed.append(
            current_fragment
            if current_count == 1
            else f"{current_fragment}__x{current_count}"
        )
        current_fragment = fragment
        current_count = 1

    collapsed.append(
        current_fragment
        if current_count == 1
        else f"{current_fragment}__x{current_count}"
    )
    return collapsed


def _compact_output_job_name(job_name: str, *, max_chars: int = 200) -> str:
    if len(job_name) <= max_chars:
        return job_name

    import hashlib

    digest = hashlib.sha1(job_name.encode("utf-8")).hexdigest()[:12]
    suffix = f"__{digest}"
    prefix = job_name[: max_chars - len(suffix)].rstrip("_.-")
    if not prefix:
        return f"job{suffix}"
    return f"{prefix}{suffix}"


def _normalise_json_regions(regions: object) -> str | None:
    if not isinstance(regions, list) or not regions:
        return None

    parts: list[str] = []
    for region in regions:
        if not isinstance(region, (tuple, list)) or len(region) != 2:
            return None
        start, end = region
        parts.append(f"{start}-{end}")
    return "_".join(parts)


def build_af3_combined_json_job_name(
    json_inputs: Sequence[dict[str, object]],
) -> str:
    fragments: list[str] = []

    for json_input in json_inputs:
        json_input_path = json_input.get("json_input")
        if not isinstance(json_input_path, str) or not json_input_path:
            continue

        fragment = _json_input_basename(json_input_path)
        region_fragment = _normalise_json_regions(json_input.get("regions"))
        if region_fragment:
            fragment = f"{fragment}__{region_fragment}"
        fragments.append(sanitise_af3_job_name(fragment))

    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return "ranked_0"
    return _compact_output_job_name(
        "_and_".join(_collapse_repeated_name_fragments(fragments))
    )


def _ensure_path_is_within_root(candidate: Path, output_root: Path) -> None:
    try:
        candidate.resolve(strict=False).relative_to(output_root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(
            f"Resolved AF3 output directory {candidate} escapes configured root {output_root}"
        ) from exc


def resolve_af3_combined_json_output_dir(
    json_inputs: Sequence[dict[str, object]],
    output_dir: str,
    *,
    use_ap_style: bool,
) -> str:
    if not use_ap_style:
        return output_dir

    output_root = Path(output_dir)
    candidate = output_root / build_af3_combined_json_job_name(json_inputs)
    _ensure_path_is_within_root(candidate, output_root)
    return os.fspath(candidate)


def resolve_af3_json_output_dir(
    json_input_path: str,
    output_dir: str,
    *,
    use_ap_style: bool,
    shared_output_root: bool,
) -> str:
    """Return the output directory for a JSON AF3 job without breaking per-job paths."""
    if not use_ap_style or not shared_output_root:
        return output_dir

    output_root = Path(output_dir)
    candidate = output_root / derive_af3_job_name_from_json(json_input_path)
    _ensure_path_is_within_root(candidate, output_root)
    return os.fspath(candidate)
