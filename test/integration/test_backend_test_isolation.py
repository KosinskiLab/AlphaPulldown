"""Backend stubs must not replace real dependencies in subsequent tests."""

import os
import subprocess
import sys
from pathlib import Path


def test_af3_backend_stubs_do_not_leak_into_modelcif_tests():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-n", "0",
            "test/unit/test_alphafold3_backend_helpers.py::"
            "test_output_name_helpers_compact_and_normalise_fragments",
            "test/unit/test_af3_modelcif.py::"
            "test_augment_real_af3_modelcif_preserves_comments_and_is_modelcif_readable",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
