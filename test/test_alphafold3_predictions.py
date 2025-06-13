#!/usr/bin/env python
"""
Unified wrapper: runs the parameterised Alphapulldown test-suite for AlphaFold3 on

  • Slurm (GPU partition)  – submits one sbatch per test-case, or
  • a local single-GPU machine – executes synchronously via pytest.

Both paths propagate the same JAX memory-control environment
variables so desktop and cluster behave identically.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from absl.testing import absltest, parameterized

# Set environment variables for GPU usage
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["JAX_FLASH_ATTENTION_IMPL"] = "xla"
# Remove deprecated variable
if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
    del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

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
    """Return environment variables for GPU usage."""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
    env["JAX_FLASH_ATTENTION_IMPL"] = "xla"
    # Remove deprecated variable if present
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
        del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
    return env


# --------------------------------------------------------------------------- #
#                                test-suite                                   #
# --------------------------------------------------------------------------- #
class TestAlphaFold3PredictStructure(parameterized.TestCase):
    # collect (job_id, log_file, test_name) for later inspection
    job_info_list: list[dict[str, str]] = []
    # Queue to collect failures
    failure_queue: Queue = Queue()
    # Dictionary to track job status
    job_status: Dict[str, str] = {}

    def _check_log_file(self, job_info: Dict[str, str], timeout: int = 7200):
        """Check a single log file for completion and PASSED status."""
        start_time = time.time()
        log_path = Path(job_info["log_file"])
        test_name = job_info["test_name"]
        
        while time.time() - start_time < timeout:
            if not log_path.exists():
                time.sleep(10)
                continue
                
            try:
                content = log_path.read_text()
                if "PASSED" in content:
                    self.job_status[test_name] = "PASSED"
                    print(f"{test_name}: PASSED")
                    return
                elif "FAILED" in content or "ERROR" in content:
                    self.job_status[test_name] = "FAILED"
                    print(f"\n--- LOG FOR FAILED TEST: {test_name} ---\n{content}\n--- END LOG ---\n")
                    self.failure_queue.put(f"{test_name}: FAILED")
                    return
            except Exception as e:
                self.job_status[test_name] = "ERROR"
                print(f"Error reading log file {log_path}: {e}")
                self.failure_queue.put(f"{test_name}: Error reading log - {e}")
                return
                
            time.sleep(10)
            
        self.job_status[test_name] = "TIMEOUT"
        self.failure_queue.put(f"{test_name}: TIMEOUT after {timeout}s")

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
        cls.failure_queue = Queue()
        cls.job_status = {}

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls.job_info_list:  # nothing was submitted → local run
            return

        # Start a thread for each job to check its log file
        with ThreadPoolExecutor(max_workers=len(cls.job_info_list)) as executor:
            futures = {
                executor.submit(cls._check_log_file, cls, job): job["test_name"]
                for job in cls.job_info_list
            }
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                test_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error checking log for {test_name}: {e}")
                    cls.failure_queue.put(f"{test_name}: Error in log checking - {e}")

        # Collect all failures
        failures = []
        while not cls.failure_queue.empty():
            failures.append(cls.failure_queue.get())

        if failures:
            raise RuntimeError("Some Slurm tests failed:\n" + "\n".join(failures))

    # ---------- per-test set-up ------------------------------------------- #
    def setUp(self):
        super().setUp()
        if not _has_gpu():
            self.skipTest("NVIDIA GPU not detected – skipping Alphafold3 tests")

        # Check for correct conda environment
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if not conda_env or conda_env != "AlphaPulldown_alphafold3":
            self.fail(f"Tests must be run in the AlphaPulldown_alphafold3 environment. Current environment: {conda_env}")

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
        conda = os.environ.get("CONDA_DEFAULT_ENV")
        if not conda or conda != "AlphaPulldown_alphafold3":
            self.fail(f"Tests must be run in the AlphaPulldown_alphafold3 environment. Current environment: {conda}")

        cmd = [
            "bash",
            "-c",
            "eval \"$(conda shell.bash hook)\""
            f" && conda activate {conda}"
            f" && echo 'Running {cls_name}::{test_name}'"
            f" && pytest -s test/check_alphafold3_predictions.py::{cls_name}::{test_name}",
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
        if not conda or conda != "AlphaPulldown_alphafold3":
            self.fail(f"Tests must be run in the AlphaPulldown_alphafold3 environment. Current environment: {conda}")

        script = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=test_alphafold3_{idx}\n"
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
            f"export XLA_CLIENT_MEM_FRACTION=$(echo \"scale=3;$MAXRAM/$GPUMEM\" | bc)\n"
            f"export TF_FORCE_UNIFIED_MEMORY=1\n"
            f"export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
            f"export XLA_FLAGS=\"--xla_gpu_enable_triton_gemm=false\"\n"
            f"# Remove deprecated variable if present\n"
            f"unset XLA_PYTHON_CLIENT_MEM_FRACTION\n\n"
            f"echo 'Running {cls_name}::{test_name}'\n"
            f"pytest -s test/check_alphafold3_predictions.py::{cls_name}::{test_name}\n"
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
        {"testcase_name": "monomer", "i": 0, "cls": "TestAlphaFold3RunModes", "test": "test__monomer"},
        {"testcase_name": "monomer_with_rna", "i": 1, "cls": "TestAlphaFold3RunModes", "test": "test__monomer_with_rna"},
        {"testcase_name": "dimer", "i": 2, "cls": "TestAlphaFold3RunModes", "test": "test__dimer"},
        {"testcase_name": "dimer_with_rna", "i": 3, "cls": "TestAlphaFold3RunModes", "test": "test__dimer_with_rna"},
        {"testcase_name": "trimer", "i": 4, "cls": "TestAlphaFold3RunModes", "test": "test__trimer"},
        {"testcase_name": "homo_oligomer", "i": 5, "cls": "TestAlphaFold3RunModes", "test": "test__homo_oligomer"},
        {"testcase_name": "chopped_dimer", "i": 6, "cls": "TestAlphaFold3RunModes", "test": "test__chopped_dimer"},
        {"testcase_name": "long_name", "i": 7, "cls": "TestAlphaFold3RunModes", "test": "test__long_name"},
    )
    def test_predict_structure(self, i: int, cls: str, test: str):
        """Route each parameterised test either through Slurm or local run."""
        if self._is_slurm_available():
            self._submit_sbatch(i, cls, test)
            # Don't show PASSED here - wait for log check
            print(f"Submitted job for {cls}::{test}")
        else:
            print("Slurm unavailable – running locally")
            self._run_test_locally(cls, test)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    absltest.main() 