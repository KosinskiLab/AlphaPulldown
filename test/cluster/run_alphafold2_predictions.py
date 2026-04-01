#!/usr/bin/env python3
"""Submit AlphaFold2 functional tests to Slurm and summarize results.

This is a standalone wrapper for `test/cluster/check_alphafold2_predictions.py`.
It is intentionally not a pytest test module, despite the filename.

Typical usage from a login node:

    python test/cluster/run_alphafold2_predictions.py

Run only selected tests:

    python test/cluster/run_alphafold2_predictions.py -k dimer
"""

from __future__ import annotations

__test__ = False

import argparse
import dataclasses
import datetime as dt
import importlib.util
import inspect
import json
import re
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Iterable

from _pytest.mark.expression import Expression


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_FILE = REPO_ROOT / "test" / "cluster" / "check_alphafold2_predictions.py"
DEFAULT_LOG_ROOT = REPO_ROOT / "test_logs"

PASS_STATES = {"COMPLETED"}
FAIL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


@dataclasses.dataclass(slots=True)
class JobSpec:
    index: int
    nodeid: str
    slug: str
    stdout_path: Path
    stderr_path: Path
    script_path: Path
    rerun_command: str
    job_id: str | None = None
    slurm_state: str | None = None
    exit_code: str | None = None
    outcome: str | None = None
    reason: str | None = None


def _has_cmd(cmd: str) -> bool:
    try:
        subprocess.run(
            [cmd, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except FileNotFoundError:
        return False


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def _normalize_state(state: str | None) -> str | None:
    if not state:
        return None
    return state.split()[0].rstrip("+")


def _slugify(value: str, *, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not slug:
        slug = "test"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("._")
    return slug


def _quote(value: str) -> str:
    return shlex.quote(value)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _default_gpu_env_lines(*, cpus_per_task: int) -> list[str]:
    thread_count = max(1, min(cpus_per_task, 4))
    return [
        "export PYTHONUNBUFFERED=1",
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{thread_count}}}"',
        f'export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-{thread_count}}}"',
        f'export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-{thread_count}}}"',
        f'export TF_NUM_INTEROP_THREADS="${{TF_NUM_INTEROP_THREADS:-{thread_count}}}"',
        f'export TF_NUM_INTRAOP_THREADS="${{TF_NUM_INTRAOP_THREADS:-{thread_count}}}"',
        'export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"',
        'export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"',
        'export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"',
        'export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"',
        'export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"',
        'if [ -z "${XLA_FLAGS:-}" ]; then export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=0 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; fi',
    ]


def _relative_nodeid_prefix(test_file: Path) -> str:
    return str(test_file.resolve().relative_to(REPO_ROOT))


def _matches_k_expression(nodeid: str, k_expr: str | None) -> bool:
    if not k_expr:
        return True
    expression = Expression.compile(k_expr)
    lowered = nodeid.lower()
    return expression.evaluate(lambda token: token.lower() in lowered)


def _collect_nodeids_from_module_import(test_file: Path, k_expr: str | None) -> list[str]:
    module_name = f"_codex_collect_{test_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {test_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    prefix = _relative_nodeid_prefix(test_file)
    nodeids: list[str] = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue
        if not issubclass(cls, unittest.TestCase):
            continue
        if not cls.__name__.startswith("Test"):
            continue

        for method_name in sorted(name for name in dir(cls) if name.startswith("test")):
            nodeid = f"{prefix}::{cls.__name__}::{method_name}"
            if _matches_k_expression(nodeid, k_expr):
                nodeids.append(nodeid)
    return nodeids


def collect_nodeids(
    *,
    python_executable: str,
    test_file: Path,
    k_expr: str | None,
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "pytest",
        "-o",
        "addopts=-ra --strict-markers",
        "--collect-only",
        "-q",
        str(test_file),
    ]
    if k_expr:
        cmd.extend(["-k", k_expr])
    result = _run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "pytest collection failed.\n"
            f"Command: {' '.join(_quote(part) for part in cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    nodeids: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ".py::" not in line:
            continue
        if line.startswith("ERROR ") or line.startswith("SKIPPED "):
            continue
        nodeids.append(line)
    if nodeids:
        return nodeids

    return _collect_nodeids_from_module_import(test_file, k_expr)


def write_job_script(
    *,
    job: JobSpec,
    python_executable: str,
    use_temp_dir: bool,
    cpus_per_task: int,
) -> None:
    pytest_cmd = [
        python_executable,
        "-m",
        "pytest",
        "-o",
        "addopts=-ra --strict-markers",
        "-vv",
        "-s",
        job.nodeid,
    ]
    if use_temp_dir:
        pytest_cmd.append("--use-temp-dir")

    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            f"cd {_quote(str(REPO_ROOT))}",
            *_default_gpu_env_lines(cpus_per_task=cpus_per_task),
            "echo \"[$(date)] Running test node:\"",
            f"echo {_quote(job.nodeid)}",
            "echo \"[$(date)] Host: $(hostname)\"",
            "echo \"[$(date)] Python: $(which python || true)\"",
            " ".join(_quote(part) for part in pytest_cmd),
            "",
        ]
    )
    job.script_path.write_text(script, encoding="utf-8")
    job.script_path.chmod(0o755)


def submit_job(job: JobSpec, args: argparse.Namespace) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--job-name={args.job_name_prefix}_{job.index:03d}",
        f"--chdir={REPO_ROOT}",
        f"--output={job.stdout_path}",
        f"--error={job.stderr_path}",
        f"--time={args.time}",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
    ]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.qos:
        cmd.append(f"--qos={args.qos}")
    if args.constraint:
        cmd.append(f"--constraint={args.constraint}")
    if args.account:
        cmd.append(f"--account={args.account}")
    if args.gres:
        cmd.append(f"--gres={args.gres}")
    for extra_arg in args.extra_sbatch_arg:
        cmd.append(extra_arg)
    cmd.append(str(job.script_path))

    result = _run(cmd)
    raw_job_id = result.stdout.strip().splitlines()[-1]
    return raw_job_id.split(";", 1)[0]


def active_job_ids(job_ids: Iterable[str]) -> set[str]:
    job_ids = [job_id for job_id in job_ids if job_id]
    if not job_ids:
        return set()

    result = _run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(job_ids),
            "-o",
            "%A",
        ],
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def query_sacct(job_id: str) -> tuple[str | None, str | None]:
    if not _has_cmd("sacct"):
        return None, None

    result = _run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State,ExitCode",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None, None

    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        job_id_raw, state, exit_code = parts[:3]
        if job_id_raw == job_id:
            return _normalize_state(state), exit_code
    return None, None


def wait_for_jobs(jobs: list[JobSpec], *, poll_interval: int, timeout_seconds: int | None) -> None:
    outstanding = {job.job_id for job in jobs if job.job_id}
    start = time.monotonic()
    previous_remaining = len(outstanding)

    while outstanding:
        if timeout_seconds is not None and (time.monotonic() - start) > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {len(outstanding)} Slurm job(s): "
                + ", ".join(sorted(outstanding))
            )

        active = active_job_ids(outstanding)
        finished = outstanding - active
        if finished:
            outstanding = active

        remaining = len(outstanding)
        if remaining != previous_remaining or finished:
            done = len(jobs) - remaining
            print(f"[wait] {done}/{len(jobs)} jobs finished, {remaining} remaining", flush=True)
            previous_remaining = remaining

        if outstanding:
            time.sleep(poll_interval)


def _combined_log_text(job: JobSpec) -> str:
    parts: list[str] = []
    if job.stdout_path.exists():
        parts.append(job.stdout_path.read_text(encoding="utf-8", errors="replace"))
    if job.stderr_path.exists():
        stderr_text = job.stderr_path.read_text(encoding="utf-8", errors="replace")
        if stderr_text:
            parts.append(stderr_text)
    return "\n".join(parts)


def _extract_reason_from_log(text: str) -> str:
    patterns = [
        r"short test summary info[\s\S]*$",
        r"=+ FAILURES =+[\s\S]*$",
        r"Traceback[\s\S]*$",
        r"(?m)^E\s+.*$",
        r"(?m)^FAILED .*$",
        r"(?m)^ERROR .*$",
        r"(?m)^.*Killed.*$",
        r"(?m)^.*PASSED.*$",
        r"(?m)^.*SKIPPED.*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            snippet = match.group(0).strip()
            if len(snippet) > 1200:
                snippet = snippet[-1200:]
            return snippet

    non_empty_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return "No log output captured."
    return "\n".join(non_empty_lines[-20:])


def classify_job(job: JobSpec) -> None:
    job.slurm_state, job.exit_code = query_sacct(job.job_id or "")
    text = _combined_log_text(job)
    state = job.slurm_state

    if state in FAIL_STATES:
        job.outcome = "FAILED"
        job.reason = f"Slurm state: {state}\n{_extract_reason_from_log(text)}"
        return

    if re.search(r"(?im)\bkilled\b", text):
        job.outcome = "FAILED"
        job.reason = _extract_reason_from_log(text)
        return

    if re.search(r"(?m)^FAILED ", text) or re.search(r"(?m)^ERROR ", text):
        job.outcome = "FAILED"
        job.reason = _extract_reason_from_log(text)
        return

    if re.search(r"=+ FAILURES =+", text) or "Traceback" in text:
        job.outcome = "FAILED"
        job.reason = _extract_reason_from_log(text)
        return

    if re.search(r"(?i)\b\d+\s+skipped\b", text) or " SKIPPED" in text:
        job.outcome = "SKIPPED"
        job.reason = _extract_reason_from_log(text)
        return

    if re.search(r"(?i)\b\d+\s+passed\b", text) or " PASSED" in text:
        job.outcome = "PASSED"
        job.reason = _extract_reason_from_log(text)
        return

    if state in PASS_STATES:
        job.outcome = "PASSED"
        job.reason = _extract_reason_from_log(text)
        return

    job.outcome = "UNKNOWN"
    job.reason = _extract_reason_from_log(text)


def write_summary(log_dir: Path, jobs: list[JobSpec]) -> Path:
    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "jobs": [
            {
                "index": job.index,
                "nodeid": job.nodeid,
                "job_id": job.job_id,
                "slurm_state": job.slurm_state,
                "exit_code": job.exit_code,
                "outcome": job.outcome,
                "stdout_log": str(job.stdout_path),
                "stderr_log": str(job.stderr_path),
                "rerun_command": job.rerun_command,
                "reason": job.reason,
            }
            for job in jobs
        ],
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def print_summary(jobs: list[JobSpec], summary_path: Path) -> int:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.outcome or "UNKNOWN"] = counts.get(job.outcome or "UNKNOWN", 0) + 1

    print("\nSummary")
    for outcome in sorted(counts):
        print(f"  {outcome}: {counts[outcome]}")
    print(f"  summary_json: {summary_path}")

    problem_jobs = [job for job in jobs if job.outcome not in {"PASSED", "SKIPPED"}]
    if problem_jobs:
        print("\nProblems")
        for job in problem_jobs:
            print(f"  {job.nodeid}")
            print(f"    slurm_job: {job.job_id}")
            print(f"    state: {job.slurm_state or 'unknown'}")
            print(f"    stdout: {job.stdout_path}")
            print(f"    stderr: {job.stderr_path}")
            print(f"    rerun: {job.rerun_command}")
            if job.reason:
                for line in job.reason.splitlines()[:20]:
                    print(f"    {line}")
    return 1 if problem_jobs else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit AlphaFold2 functional tests to Slurm in parallel, wait for completion, "
            "and summarize the logs."
        )
    )
    parser.add_argument(
        "nodeid",
        nargs="*",
        help=(
            "Optional exact pytest node IDs to submit. If omitted, tests are collected "
            f"from {DEFAULT_TEST_FILE.relative_to(REPO_ROOT)}."
        ),
    )
    parser.add_argument(
        "--test-file",
        default=str(DEFAULT_TEST_FILE),
        help="Pytest file to collect from. Defaults to test/cluster/check_alphafold2_predictions.py",
    )
    parser.add_argument(
        "-k",
        dest="k_expr",
        default=None,
        help="Optional pytest -k expression applied during collection.",
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help="Submit at most this many collected tests.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List collected node IDs and exit without submitting jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect tests and write job scripts, but do not call sbatch.",
    )
    parser.add_argument(
        "--use-temp-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run target tests with isolated temporary output directories. "
            "Use --no-use-temp-dir to keep the shared repo output tree."
        ),
    )
    parser.add_argument("--partition", default="gpu-el8", help="Slurm partition/queue.")
    parser.add_argument("--qos", default="normal", help="Slurm QoS.")
    parser.add_argument("--constraint", default="gaming", help="Optional Slurm constraint.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres request, for example gpu:1.")
    parser.add_argument("--time", default="12:00:00", help="Per-job walltime.")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="CPUs per Slurm task.")
    parser.add_argument("--mem", default="16G", help="Per-job memory request.")
    parser.add_argument(
        "--extra-sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch argument. Can be passed multiple times.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="af2test",
        help="Prefix for Slurm job names.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between Slurm polling cycles.",
    )
    parser.add_argument(
        "--wait-timeout-hours",
        type=float,
        default=24.0,
        help="Maximum hours to wait for all submitted jobs. Use 0 to disable.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write job scripts and logs into. Defaults to test_logs/alphafold2_<timestamp>.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used both for collection and inside Slurm jobs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _has_cmd("sbatch") and not args.list and not args.dry_run:
        raise SystemExit("sbatch is not available in PATH.")
    if not _has_cmd("squeue") and not args.list and not args.dry_run:
        raise SystemExit("squeue is not available in PATH.")

    test_file = Path(args.test_file).resolve()
    if not test_file.exists():
        raise SystemExit(f"Test file does not exist: {test_file}")

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_dir = (DEFAULT_LOG_ROOT / f"alphafold2_{_timestamp()}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.nodeid:
        nodeids = list(args.nodeid)
    else:
        nodeids = collect_nodeids(
            python_executable=args.python,
            test_file=test_file,
            k_expr=args.k_expr,
        )

    if args.max_tests is not None:
        nodeids = nodeids[: args.max_tests]

    if not nodeids:
        print("No tests matched the requested selection.")
        return 0

    if not args.use_temp_dir and len(nodeids) > 1:
        raise SystemExit(
            "--no-use-temp-dir is not safe for parallel AF2 wrapper runs because "
            "the tests share and clean common output roots. Re-run with the default "
            "--use-temp-dir, or submit a single nodeid at a time."
        )

    if args.list:
        for nodeid in nodeids:
            print(nodeid)
        return 0

    print(f"Collected {len(nodeids)} test node(s).")
    print(f"Log directory: {log_dir}")

    jobs: list[JobSpec] = []
    for index, nodeid in enumerate(nodeids, start=1):
        slug = _slugify(nodeid)
        stdout_path = log_dir / f"{index:03d}_{slug}.out"
        stderr_path = log_dir / f"{index:03d}_{slug}.err"
        script_path = log_dir / f"{index:03d}_{slug}.sbatch.sh"
        rerun_command = (
            f"{_quote(args.python)} -m pytest -o {_quote('addopts=-ra --strict-markers')} -vv -s {_quote(nodeid)}"
            + (" --use-temp-dir" if args.use_temp_dir else "")
        )
        job = JobSpec(
            index=index,
            nodeid=nodeid,
            slug=slug,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            script_path=script_path,
            rerun_command=rerun_command,
        )
        write_job_script(
            job=job,
            python_executable=args.python,
            use_temp_dir=args.use_temp_dir,
            cpus_per_task=args.cpus_per_task,
        )
        jobs.append(job)

    if args.dry_run:
        print("Dry run only. Prepared job scripts:")
        for job in jobs:
            print(f"  {job.nodeid}")
            print(f"    script: {job.script_path}")
            print(f"    stdout: {job.stdout_path}")
            print(f"    stderr: {job.stderr_path}")
        return 0

    for job in jobs:
        job.job_id = submit_job(job, args)
        print(f"[submit] {job.job_id}  {job.nodeid}")

    timeout_seconds: int | None
    if args.wait_timeout_hours <= 0:
        timeout_seconds = None
    else:
        timeout_seconds = int(args.wait_timeout_hours * 3600)

    wait_for_jobs(
        jobs,
        poll_interval=args.poll_interval,
        timeout_seconds=timeout_seconds,
    )

    for job in jobs:
        classify_job(job)

    summary_path = write_summary(log_dir, jobs)
    return print_summary(jobs, summary_path)


if __name__ == "__main__":
    raise SystemExit(main())
