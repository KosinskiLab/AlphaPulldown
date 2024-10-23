import logging
import subprocess
import os
import time
from absl.testing import absltest, parameterized
import shutil
import re

"""
Wrapper for running structure prediction tests
"""

class TestPredictStructure(parameterized.TestCase):
    # Class-level list to store job information
    job_info_list = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        t = time.localtime()
        # Include minutes and seconds to ensure uniqueness
        cls.base_path = f"slurm_logs/{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}_{t.tm_min:02d}_{t.tm_sec:02d}"
        if os.path.exists(cls.base_path):
            logging.warning("Warning: slurm_logs directory already exists. Overwriting log files...")
            shutil.rmtree(cls.base_path)
        os.makedirs(cls.base_path)

    def setUp(self) -> None:
        super().setUp()
        # Unique directory for each test
        self.path = os.path.join(self.base_path, self._testMethodName)
        os.makedirs(self.path, exist_ok=True)

    def _is_slurm_available(self):
        try:
            subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _run_test_locally(self, class_name: str, test_name: str, conda_env: str):
        # Synchronous execution for local tests
        command = [
            "bash", "-c",
            f"""
            eval "$(conda shell.bash hook)"
            conda activate {conda_env}
            echo "Running {class_name}::{test_name}"
            pytest -s test/check_predict_structure.py::{class_name}::{test_name}
            """
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()

    def _extract_job_id(self, submission_output: str) -> str:
        # Extract the job ID from sbatch output
        match = re.search(r'Submitted batch job (\d+)', submission_output)
        if match:
            return match.group(1)
        else:
            self.fail(f"Failed to extract job ID from sbatch output: {submission_output}")

    def _sbatch_command(self, i: int, class_name: str, test_name: str):
        if not self._is_slurm_available():
            self.skipTest("Slurm not available")

        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env is None:
            self.fail("CONDA_DEFAULT_ENV not set")

        script_content = f"""#!/bin/bash
#SBATCH --job-name=test_predict_structure_{i}
#SBATCH --time=02:00:00
#SBATCH --qos=normal
#SBATCH -p gpu-el8
#SBATCH -C gaming
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000

eval "$(conda shell.bash hook)"
conda activate {conda_env}

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1)
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=3;$MAXRAM / $GPUMEM"|bc)
export TF_FORCE_UNIFIED_MEMORY='1'
echo "Running {class_name}::{test_name}"
pytest -s test/check_predict_structure.py::{class_name}::{test_name}
        """

        script_path = f"{self.path}/test_{i}_{class_name}_{test_name}.sh"
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)

        log_file = f"{self.path}/test_{i}_{class_name}_{test_name}.log"
        command = [
            "sbatch",
            f"--output={log_file}",
            script_path
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        job_submission_output = result.stdout.strip()
        print(job_submission_output)
        job_id = self._extract_job_id(job_submission_output)
        # Store job info for later tracking
        self.__class__.job_info_list.append({
            'job_id': job_id,
            'log_file': log_file,
            'test_name': f"{class_name}::{test_name}"
        })

    @parameterized.named_parameters(
        {"testcase_name": "monomer", "i": 0, "class_name": "TestRunModes", "test_name": "test__monomer"},
        {"testcase_name": "dimer", "i": 1, "class_name": "TestRunModes", "test_name": "test__dimer"},
        {"testcase_name": "trimer", "i": 2, "class_name": "TestRunModes", "test_name": "test__trimer"},
        {"testcase_name": "chopped_dimer", "i": 3, "class_name": "TestRunModes", "test_name": "test__chopped_dimer"},
        {"testcase_name": "homo_oligomer", "i": 4, "class_name": "TestRunModes", "test_name": "test__homo_oligomer"},
        {"testcase_name": "no_relax", "i": 5, "class_name": "TestResume", "test_name": "test__no_relax"},
        {"testcase_name": "relax_all", "i": 6, "class_name": "TestResume", "test_name": "test__relax_all"},
        {"testcase_name": "continue_relax", "i": 7, "class_name": "TestResume", "test_name": "test__continue_relax"},
        {"testcase_name": "continue_prediction", "i": 8, "class_name": "TestResume", "test_name": "test__continue_prediction"},
        {"testcase_name": "long_name", "i": 9, "class_name": "TestRunModes", "test_name": "test__long_name"},
    )
    def test_predict_structure(self, i: int, class_name: str, test_name: str):
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not self._is_slurm_available():
            print("Slurm not available, running test locally")
            self._run_test_locally(class_name, test_name, conda_env)
        else:
            # Submit the job without waiting
            self._sbatch_command(i, class_name, test_name)
            # Do not wait here; proceed to submit the next job

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls.job_info_list:
            # No jobs were submitted
            return

        # Wait for all jobs to complete
        all_jobs_completed = False
        timeout = 7200  # Timeout in seconds (e.g., 2 hours)
        start_time = time.time()

        job_ids = [job_info['job_id'] for job_info in cls.job_info_list]
        job_id_set = set(job_ids)

        while not all_jobs_completed:
            try:
                result = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                running_job_ids = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
                pending_jobs = job_id_set & running_job_ids
                if not pending_jobs:
                    all_jobs_completed = True
                    break
            except Exception as e:
                print(f"Failed to check job status: {e}")
                break  # Avoid infinite loop on error

            if time.time() - start_time > timeout:
                print(f"Jobs did not complete within {timeout} seconds.")
                break

            time.sleep(10)  # Wait before checking again

        # After all jobs have completed, check their logs
        failed_tests = []
        for job_info in cls.job_info_list:
            log_file = job_info['log_file']
            test_name = job_info['test_name']
            if not os.path.exists(log_file):
                failed_tests.append(f"{test_name}: Log file {log_file} does not exist.")
                continue
            with open(log_file, 'r') as f:
                log_content = f.read()
            if 'PASSED' in log_content:
                print(f"{test_name}: PASSED")
            else:
                failed_tests.append(f"{test_name}: FAILED. 'PASSED' not found in log.")

        if failed_tests:
            # Report failures collectively
            failure_message = '\n'.join(failed_tests)
            raise Exception(f"Some tests failed:\n{failure_message}")

if __name__ == '__main__':
    absltest.main()
