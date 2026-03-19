import json
import os
import string
from pathlib import Path


_AF3_ALLOWED_NAME_CHARS = set(string.ascii_lowercase + string.digits + "_-.")


def sanitise_af3_job_name(job_name: str) -> str:
    """Match AlphaFold 3's filename sanitisation for job names."""
    lower_spaceless_name = job_name.lower().replace(" ", "_")
    sanitised = "".join(ch for ch in lower_spaceless_name if ch in _AF3_ALLOWED_NAME_CHARS)
    return sanitised or "ranked_0"


def derive_af3_job_name_from_json(json_input_path: str) -> str:
    """Derive the AF3 job name from the current JSON input."""
    fallback_name = Path(json_input_path).stem

    try:
        with open(json_input_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return sanitise_af3_job_name(fallback_name)

    if isinstance(payload, dict):
        raw_name = payload.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return sanitise_af3_job_name(raw_name)

    return sanitise_af3_job_name(fallback_name)


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

    return os.path.join(output_dir, derive_af3_job_name_from_json(json_input_path))
