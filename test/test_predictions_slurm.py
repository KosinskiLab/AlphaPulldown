import logging
import subprocess
import os
import time
from absl.testing import absltest, parameterized
import shutil
import re

"""
Wrapper for running structure prediction tests with Slurm or locally.
We define a parameterized set of 20 testcases (10 AF2 + 10 AF3).
"""

class TestPredictStructure(parameterized.TestCase):
    job_info_list = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        t = time.localtime()
        cls.base_path = f"slurm_logs/{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday}_{t.tm_hour:02d}_{t.tm_min:02d}_{t.tm_sec:02d}"
        if os.path.exists(cls.base_path):
            logging.warning("Warning: slurm_logs directory already exists. Overwriting log files...")
            shutil.rmtree(cls.base_path)
        os.makedirs(cls.base_path)

    def setUp(self) -> None:
        super().setUp()
        self.path = os.path.join(self.base_path, self._testMethodName)
        os.makedirs(self.path, exist_ok=True)

    def _is_slurm_available(self):
        try:
            subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _run_test_locally(self, class_name: str, method_name: str, conda_env: str):
        cmd = [
            "bash", "-c",
            f"""
            eval "$(conda shell.bash hook)"
            conda activate {conda_env}
            echo "Running {class_name}::{method_name}"
            pytest -s test/check_predict_structure.py::{class_name}::{method_name}
            """
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()

    def _extract_job_id(self, submission_output: str) -> str:
        match = re.search(r'Submitted batch job (\d+)', submission_output)
        if match:
            return match.group(1)
        else:
            self.fail(f"Failed to extract job ID from sbatch output: {submission_output}")

    def _sbatch_command(self, i: int, class_name: str, method_name: str):
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

echo "Running {class_name}::{method_name}"
pytest -s test/check_predict_structure.py::{class_name}::{method_name}
"""

        script_path = f"{self.path}/test_{i}_{class_name}_{method_name}.sh"
        with open(script_path, 'w') as sf:
            sf.write(script_content)

        log_file = f"{self.path}/test_{i}_{class_name}_{method_name}.log"
        cmd = [
            "sbatch",
            f"--output={log_file}",
            script_path
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
        submission_out = result.stdout.strip()
        print(submission_out)
        job_id = self._extract_job_id(submission_out)
        self.__class__.job_info_list.append({
            'job_id': job_id,
            'log_file': log_file,
            'test_name': f"{class_name}::{method_name}"
        })

    @parameterized.named_parameters(
        # AF2
        {"testcase_name": "00_monomer",             "i":  0, "class_name": "TestRunModes", "method_name": "test__monomer"},
        {"testcase_name": "01_dimer",               "i":  1, "class_name": "TestRunModes", "method_name": "test__dimer"},
        {"testcase_name": "02_trimer",              "i":  2, "class_name": "TestRunModes", "method_name": "test__trimer"},
        {"testcase_name": "03_chopped_dimer",       "i":  3, "class_name": "TestRunModes", "method_name": "test__chopped_dimer"},
        {"testcase_name": "04_homo_oligomer",       "i":  4, "class_name": "TestRunModes", "method_name": "test__homo_oligomer"},
        {"testcase_name": "05_no_relax",            "i":  5, "class_name": "TestResume",   "method_name": "test__no_relax"},
        {"testcase_name": "06_relax_all",           "i":  6, "class_name": "TestResume",   "method_name": "test__relax_all"},
        {"testcase_name": "07_continue_relax",      "i":  7, "class_name": "TestResume",   "method_name": "test__continue_relax"},
        {"testcase_name": "08_continue_prediction", "i":  8, "class_name": "TestResume",   "method_name": "test__continue_prediction"},
        {"testcase_name": "09_long_name",           "i":  9, "class_name": "TestRunModes", "method_name": "test__long_name"},

        # AF3
        {"testcase_name": "10_monomer_af3",             "i": 10, "class_name": "TestRunModes", "method_name": "test__monomer_af3"},
        {"testcase_name": "11_dimer_af3",               "i": 11, "class_name": "TestRunModes", "method_name": "test__dimer_af3"},
        {"testcase_name": "12_trimer_af3",              "i": 12, "class_name": "TestRunModes", "method_name": "test__trimer_af3"},
        {"testcase_name": "13_chopped_dimer_af3",       "i": 13, "class_name": "TestRunModes", "method_name": "test__chopped_dimer_af3"},
        {"testcase_name": "14_homo_oligomer_af3",       "i": 14, "class_name": "TestRunModes", "method_name": "test__homo_oligomer_af3"},
        {"testcase_name": "15_no_relax_af3",            "i": 15, "class_name": "TestResume",   "method_name": "test__no_relax_af3"},
        {"testcase_name": "16_relax_all_af3",           "i": 16, "class_name": "TestResume",   "method_name": "test__relax_all_af3"},
        {"testcase_name": "17_continue_relax_af3",      "i": 17, "class_name": "TestResume",   "method_name": "test__continue_relax_af3"},
        {"testcase_name": "18_continue_prediction_af3", "i": 18, "class_name": "TestResume",   "method_name": "test__continue_prediction_af3"},
        {"testcase_name": "19_long_name_af3",           "i": 19, "class_name": "TestRunModes", "method_name": "test__long_name_af3"},
    )
    def test_predict_structure(self, i: int, class_name: str, method_name: str):
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not self._is_slurm_available():
            print("Slurm not available, running test locally")
            self._run_test_locally(class_name, method_name, conda_env)
        else:
            self._sbatch_command(i, class_name, method_name)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls.job_info_list:
            return

        # Wait for all Slurm jobs to complete, up to 2 hours
        all_jobs_completed = False
        timeout = 7200  # 2 hours
        start_time = time.time()
        job_ids = [job_info['job_id'] for job_info in cls.job_info_list]
        job_id_set = set(job_ids)

        while not all_jobs_completed:
            try:
                sq = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                running_job_ids = set(sq.stdout.strip().split('\n')) if sq.stdout.strip() else set()
                pending = job_id_set & running_job_ids
                if not pending:
                    all_jobs_completed = True
                    break
            except Exception as e:
                print(f"Failed to check job status: {e}")
                break

            if time.time() - start_time > timeout:
                print(f"Jobs did not complete within {timeout} seconds.")
                break
            time.sleep(10)

        # After all jobs have completed, check logs
        failed_tests = []
        for job_info in cls.job_info_list:
            lf = job_info['log_file']
            tn = job_info['test_name']
            if not os.path.exists(lf):
                failed_tests.append(f"{tn}: Log file {lf} does not exist.")
                continue
            with open(lf, 'r') as f:
                log_content = f.read()
            if 'PASSED' in log_content:
                print(f"{tn}: PASSED")
            else:
                failed_tests.append(f"{tn}: FAILED. 'PASSED' not found in log.")

        if failed_tests:
            msg = '\n'.join(failed_tests)
            raise Exception(f"Some tests failed:\n{msg}")


if __name__ == '__main__':
    absltest.main()
