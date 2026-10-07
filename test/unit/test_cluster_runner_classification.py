"""The GPU test runners judge a job by pytest's own result line, not by tracebacks a passing run happens to log."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "test" / "cluster" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses look their module up while the class is created
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["run_alphafold2_predictions", "run_alphafold3_predictions"])
def runner(request):
    return _load(request.param)


def _classify(runner, monkeypatch, text, state="COMPLETED"):
    monkeypatch.setattr(runner, "query_sacct", lambda job_id: (state, "0:0"))
    monkeypatch.setattr(runner, "_combined_log_text", lambda job: text)
    job = runner.JobSpec(index=1, nodeid="t::x", slug="x", stdout_path=Path("o"), stderr_path=Path("e"),
                         script_path=Path("s"), rerun_command="", job_id="1")
    runner.classify_job(job)
    return job.outcome


LOGGED_TRACEBACK = "Traceback (most recent call last):\npydantic_core.ValidationError: ...\n"


def test_logged_traceback_does_not_fail_a_passing_test(runner, monkeypatch):
    text = LOGGED_TRACEBACK + "PASSED\n\n======== 1 passed in 95.36s (0:01:35) ========\n"
    assert _classify(runner, monkeypatch, text) == "PASSED"


@pytest.mark.parametrize(
    "line, outcome",
    [("== 1 failed in 3.1s ==", "FAILED"), ("== 1 passed, 1 error in 3.1s ==", "FAILED"),
     ("== 1 skipped in 0.5s ==", "SKIPPED"), ("== 2 passed, 1 skipped in 9.0s ==", "PASSED")],
)
def test_pytest_result_line_decides(runner, monkeypatch, line, outcome):
    assert _classify(runner, monkeypatch, LOGGED_TRACEBACK + line + "\n") == outcome


def test_without_a_result_line_a_traceback_still_fails(runner, monkeypatch):
    assert _classify(runner, monkeypatch, LOGGED_TRACEBACK) == "FAILED"


def test_a_failed_slurm_job_fails_whatever_the_log_says(runner, monkeypatch):
    assert _classify(runner, monkeypatch, "== 1 passed in 1.0s ==\n", state="FAILED") == "FAILED"
