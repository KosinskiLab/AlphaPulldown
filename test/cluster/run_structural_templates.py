#!/usr/bin/env python3
"""Submit the structural-template GPU smoke test to Slurm.

Standalone wrapper for `test/cluster/check_structural_templates.py`, and
intentionally not a pytest module despite the filename.

The submission machinery -- collect node IDs, write one sbatch script per test,
wait, classify the outcomes, write a summary -- already exists in
`run_alphafold2_predictions.py` and takes a ``--test-file``. Copying it here
would mean maintaining a third near-identical copy of it, so this wrapper only
supplies different defaults and delegates. Every option that submitter accepts
(``--partition``, ``--gres``, ``--constraint``, ``--time``, ``--mem``,
``--dry-run``, ``--list``, ``-k`` ...) works here unchanged, and anything passed
on the command line wins over the defaults below.

Typical usage from a login node:

    python test/cluster/run_structural_templates.py

Check what would be submitted, without submitting:

    python test/cluster/run_structural_templates.py --list

The test itself skips unless FOLDSEEK_BINARY / FOLDSEEK_DATABASE_PATH /
ESMFOLD_MODEL_DIR / ALPHAFOLD_DATA_DIR are set and a GPU is present; see the
module docstring of check_structural_templates.py.
"""

from __future__ import annotations

__test__ = False

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_FILE = REPO_ROOT / "test" / "cluster" / "check_structural_templates.py"
SUBMITTER = REPO_ROOT / "test" / "cluster" / "run_alphafold2_predictions.py"

# ESMFold's weights and activations are the memory story here; Foldseek and the
# featuriser are comparatively small, and the smoke test folds one short chain.
DEFAULT_OPTIONS = {
    "--test-file": str(CHECK_FILE),
    "--mem": "48G",
    "--time": "04:00:00",
    "--cpus-per-task": "8",
}


def _load_submitter():
    specification = importlib.util.spec_from_file_location(
        "alphapulldown_cluster_submitter", SUBMITTER
    )
    if specification is None or specification.loader is None:
        raise SystemExit(f"Cannot load the Slurm submitter from {SUBMITTER}")
    module = importlib.util.module_from_spec(specification)
    # Registered before execution because the submitter defines slotted
    # dataclasses, and dataclasses resolves their annotations through
    # sys.modules[cls.__module__].
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _with_defaults(argv: list[str]) -> list[str]:
    """Apply the defaults the caller did not already set."""
    arguments = list(argv)
    for option, value in DEFAULT_OPTIONS.items():
        already_given = any(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
        )
        if not already_given:
            arguments.extend([option, value])
    return arguments


def main() -> int:
    if not CHECK_FILE.is_file():
        raise SystemExit(f"Test file does not exist: {CHECK_FILE}")
    submitter = _load_submitter()
    sys.argv = [sys.argv[0], *_with_defaults(sys.argv[1:])]
    return submitter.main()


if __name__ == "__main__":
    raise SystemExit(main())
