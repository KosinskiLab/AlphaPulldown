"""Run the most expensive GPU tests only when the code they cover has changed.

A test that covers some code runs when the checkout differs from its merge base with
origin/main in that code, committed, uncommitted or untracked. On main itself nothing
differs, so release validation sets RUN_EXPENSIVE_GPU_TESTS=1 to run them anyway. If git
cannot tell what changed, they run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORCE = "RUN_EXPENSIVE_GPU_TESTS"

# Feature generation, its metadata and the ModelCIF files that carry it; the submodules
# hold the AlphaFold 2 and 3 data pipelines.
FEATURE_GENERATION = (
    "alphapulldown/features",
    "alphapulldown/scripts/create_individual_features.py",
    "alphapulldown/objects.py",
    "alphapulldown/folding_backend/alphafold3_modelcif.py",
    "alphapulldown/utils/create_custom_template_db.py",
    "alphapulldown/utils/feature_metadata.py",
    "alphapulldown/utils/file_handling.py",
    "alphapulldown/utils/mmseqs_species_identifiers.py",
    "alphapulldown/utils/modelling_setup.py",
    "alphapulldown/utils/msa_formats.py",
    "alphapulldown/utils/multimeric_template_utils.py",
    "alphapulldown/utils/sequence_types.py",
    "alphapulldown/utils/template_reuse.py",
    "alphafold",
    "alphafold3",
)
# The fused AlphaFold 2-Multimer kernels: AlphaPulldown's policy and ColabFold's kernels.
FAST_KERNELS = (
    "alphapulldown/prediction/fast_kernels.py",
    "ColabFold",
    "alphafold",
)


def changed_paths(base: str = "origin/main") -> set[str] | None:
    """Repository paths that differ from the merge base with ``base``; None if unknown."""

    def git(*args: str) -> list[str]:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.split()

    try:
        (merge_base,) = git("merge-base", "HEAD", base)
        return set(git("diff", "--name-only", merge_base)) | set(
            git("ls-files", "--others", "--exclude-standard")
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def skip_unless_changed(covered: tuple[str, ...], base: str = "origin/main") -> str | None:
    """Why to skip a test of the ``covered`` code, or None to run it."""
    if os.getenv(FORCE, "").strip().lower() in ("1", "true", "yes"):
        return None
    changed = changed_paths(base)
    if changed is None:
        return None
    if any(path == root or path.startswith(f"{root}/") for path in changed for root in covered):
        return None
    return (
        f"expensive, and none of the code it covers differs from {base}; "
        f"set {FORCE}=1 to run it anyway"
    )
