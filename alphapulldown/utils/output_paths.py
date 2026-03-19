import json
import os
import string
from pathlib import Path


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

    try:
        candidate.resolve(strict=False).relative_to(output_root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(
            f"Resolved AF3 output directory {candidate} escapes configured root {output_root}"
        ) from exc

    return os.fspath(candidate)
