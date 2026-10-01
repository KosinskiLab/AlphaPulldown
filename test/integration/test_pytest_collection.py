"""CI must execute every parametrized case, including a failing later case."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("parallel", [False, True])
def test_repository_hooks_preserve_parameter_identity(tmp_path, parallel):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "conftest.py").write_text((root / "conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_parameters.py").write_text(
        'import pytest\n'
        '@pytest.mark.parametrize("value", [0, 1])\n'
        'def test_value(value):\n'
        '    assert value == 0\n'
    )
    command = [sys.executable, "-m", "pytest", "-q", "--tb=no"]
    if parallel:
        command += ["-n", "2", "--dist", "loadfile"]
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        command, cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed" in result.stdout
