#!/usr/bin/env python
"""
Unified wrapper: runs the parameterised Alphapulldown test-suite on

  • Slurm (GPU partition)  – submits one sbatch per test-case, or
  • a local single-GPU machine – executes synchronously via pytest.

Both paths propagate the same JAX/TF memory-control environment
variables so desktop and cluster behave identically.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from absl.testing import absltest, parameterized


# --------------------------------------------------------------------------- #
#                          generic capability helpers                         #
# --------------------------------------------------------------------------- #
def _has_cmd(cmd: str) -> bool:
    """True iff *cmd* exists in PATH and exits with status 0."""
    try:
        subprocess.run([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _has_gpu() -> bool:
    return _has_cmd("nvidia-smi")


def _gpu_env() -> dict[str, str]:
    """
    Return a copy of os.environ with extra keys that stop XLA/JAX from
    pre-allocating all GPU RAM.  Works the same in Slurm and local mode.
    """
    env = os.environ.copy()
    try:
        total = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode()
            .splitlines()[0]
        )
        # leave ≈10 % head-room
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{max(0.1, min(0.9, 0.9 * 16000 / total)):.3f}"
    except Exception:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    env["TF_FORCE_UNIFIED_MEMORY"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return env


# --------------------------------------------------------------------------- #
#                                test-suite                                   #
# --------------------------------------------------------------------------- #
class TestPredictStructure(parameterized.TestCase):
    # collect (job_id, log_file, test_name) for later inspection
    job_info_list: list[dict[str, str]] = []

    # ---------- suite-level set-up / tear-down ----------------------------- #
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        t = time.localtime()
        cls.base_path = Path(
            f"test_logs/{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
            f"_{t.tm_hour:02d}_{t.tm_min:02d}_{t.tm_sec:02d}"
        )
        if cls.base_path.exists():
            logging.warning("test_logs directory already exists – deleting it")
            shutil.rmtree(cls.base_path)
        cls.base_path.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls.job_info_list:  # nothing was submitted → local run
            return

        # -------- wait until all Slurm jobs are gone ---------------------- #
        job_ids = {j["job_id"] for j in cls.job_info_list}
        timeout = 7200  # seconds
        start = time.time()

        while True:
            try:
                running = set(
                    subprocess.check_output(
                        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%A"]
                    )
                    .decode()
                    .strip()
                    .splitlines()
                )
            except Exception as exc:
                print(f"squeue failed: {exc}")
                break

            if not (job_ids & running):
                break
            if time.time() - start > timeout:
                print("timeout while waiting for Slurm jobs; continuing anyway")
                break
            time.sleep(10)

        # -------- scan logs for PASS/FAIL --------------------------------- #
        failures = []
        for job in cls.job_info_list:
            lf = Path(job["log_file"])
            if not lf.exists():
                failures.append(f"{job['test_name']}: log file missing")
                continue
            content = lf.read_text()
            if "PASSED" in content:
                print(f"{job['test_name']}: PASSED")
            else:
                failures.append(f"{job['test_name']}: FAILED – keyword 'PASSED' absent")

        if failures:
            raise RuntimeError("Some Slurm tests failed:\n" + "\n".join(failures))

    # ---------- per-test set-up ------------------------------------------- #
    def setUp(self):
        super().setUp()
        if not _has_gpu():
            self.skipTest("NVIDIA GPU not detected – skipping Alphafold tests")

        self.case_dir = self.base_path / self._testMethodName
        self.case_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    #                    internal helpers (local vs Slurm)                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _is_slurm_available() -> bool:
        return _has_cmd("sinfo")

    @staticmethod
    def _extract_job_id(text: str) -> str:
        m = re.search(r"Submitted batch job (\d+)", text)
        if not m:
            raise RuntimeError(f"could not parse job-id from:\n{text}")
        return m.group(1)

    # -- run on the workstation ------------------------------------------- #
    def _run_test_locally(self, cls_name: str, test_name: str):
        conda = os.environ.get("CONDA_DEFAULT_ENV") or ""
        cmd = [
            "bash",
            "-c",
            "eval \"$(conda shell.bash hook)\""
            f" && conda activate {conda}"
            f" && echo 'Running {cls_name}::{test_name}'"
            f" && pytest -s test/check_predict_structure.py::{cls_name}::{test_name}",
        ]
        res = subprocess.run(cmd, text=True, capture_output=True, env=_gpu_env())
        print(res.stdout)
        print(res.stderr)
        res.check_returncode()

    # -- submit one sbatch script ----------------------------------------- #
    def _submit_sbatch(self, idx: int, cls_name: str, test_name: str):
        if not self._is_slurm_available():
            self.skipTest("requested Slurm path, but Slurm unavailable")

        conda = os.environ.get("CONDA_DEFAULT_ENV")
        if not conda:
            self.fail("CONDA_DEFAULT_ENV is not set inside Slurm submission")

        script = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=test_predict_structure_{idx}\n"
            f"#SBATCH --time=12:00:00\n"
            f"#SBATCH --qos=normal\n"
            f"#SBATCH -p gpu-el8\n"
            f"#SBATCH -C gaming\n"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH -N 1\n"
            f"#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task=8\n"
            f"#SBATCH --mem=16000\n\n"
            f"eval \"$(conda shell.bash hook)\"\n"
            f"conda activate {conda}\n\n"
            f"MAXRAM=$(echo $(ulimit -m) / 1024.0 | bc)\n"
            f"GPUMEM=$(nvidia-smi --query-gpu=memory.total"
            f" --format=csv,noheader,nounits | head -1)\n"
            f"export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo \"scale=3;$MAXRAM/$GPUMEM\" | bc)\n"
            f"export TF_FORCE_UNIFIED_MEMORY=1\n"
            f"export XLA_PYTHON_CLIENT_PREALLOCATE=false\n\n"
            f"echo 'Running {cls_name}::{test_name}'\n"
            f"pytest -s test/check_predict_structure.py::{cls_name}::{test_name}\n"
        )

        script_path = self.case_dir / f"test_{idx}_{cls_name}_{test_name}.sh"
        script_path.write_text(script)
        log_path = self.case_dir / f"test_{idx}_{cls_name}_{test_name}.log"

        res = subprocess.run(
            ["sbatch", f"--output={log_path}", str(script_path)],
            text=True,
            capture_output=True,
            check=True,
        )
        job_id = self._extract_job_id(res.stdout.strip())
        print(res.stdout.strip())

        self.__class__.job_info_list.append(
            {"job_id": job_id, "log_file": str(log_path), "test_name": f"{cls_name}::{test_name}"}
        )

    # --------------------------------------------------------------------- #
    #                           the actual parameterization                 #
    # --------------------------------------------------------------------- #
    @parameterized.named_parameters(
        {"testcase_name": "monomer", "i": 0, "cls": "TestRunModes", "test": "test__monomer"},
        {"testcase_name": "dimer", "i": 1, "cls": "TestRunModes", "test": "test__dimer"},
        {"testcase_name": "trimer", "i": 2, "cls": "TestRunModes", "test": "test__trimer"},
        {"testcase_name": "chopped_dimer", "i": 3, "cls": "TestRunModes", "test": "test__chopped_dimer"},
        {"testcase_name": "homo_oligomer", "i": 4, "cls": "TestRunModes", "test": "test__homo_oligomer"},
        {"testcase_name": "no_relax", "i": 5, "cls": "TestResume", "test": "test__no_relax"},
        {"testcase_name": "relax_all", "i": 6, "cls": "TestResume", "test": "test__relax_all"},
        {"testcase_name": "continue_relax", "i": 7, "cls": "TestResume", "test": "test__continue_relax"},
        {"testcase_name": "continue_prediction", "i": 8, "cls": "TestResume", "test": "test__continue_prediction"},
        {"testcase_name": "long_name", "i": 9, "cls": "TestRunModes", "test": "test__long_name"},
    )
    def test_predict_structure(self, i: int, cls: str, test: str):
        """Route each parameterised test either through Slurm or local run."""
        if self._is_slurm_available():
            self._submit_sbatch(i, cls, test)
        else:
            print("Slurm unavailable – running locally")
            self._run_test_locally(cls, test)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    absltest.main()
