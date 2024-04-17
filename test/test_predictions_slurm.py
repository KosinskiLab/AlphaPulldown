import subprocess
import os
import pytest
"""
Just to be able to run all tests with a single command
"""


def is_slurm_available():
    try:
        subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def sbatch_command(i):
    if not is_slurm_available():
        pytest.skip("Slurm not available")

    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env is None:
        pytest.fail("CONDA_DEFAULT_ENV not set")
    command = [
        "sbatch",
        f"--array={i}",
        f"--output=%j.testRun_{i}.log",
        "test/test_predict_structure.sh",
        conda_env]
    subprocess.run(command, check=True)


def test_predict_monomer():
    sbatch_command(1)

def test_without_relaxation():
    sbatch_command(2)

def test_with_relaxation_all_models():
    sbatch_command(3)

def test_resume_relaxation():
    sbatch_command(4)

def test_resume_prediction():
    sbatch_command(5)

def test_true_multimer():
    sbatch_command(6)