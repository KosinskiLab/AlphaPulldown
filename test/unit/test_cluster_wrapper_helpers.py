import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AF2_CHECK_PATH = REPO_ROOT / "test" / "cluster" / "check_alphafold2_predictions.py"
AF2_WRAPPER_PATH = REPO_ROOT / "test" / "cluster" / "run_alphafold2_predictions.py"
AF3_WRAPPER_PATH = REPO_ROOT / "test" / "cluster" / "run_alphafold3_predictions.py"


def _load_module(module_name: str, module_path: Path):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_af2_cluster_subprocess_env_sets_safe_gpu_defaults(monkeypatch):
    module = _load_module("test_cluster_af2_check_module", AF2_CHECK_PATH)

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        "TF_CPP_MIN_LOG_LEVEL",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)

    env = module._af2_subprocess_env()

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["TF_NUM_INTEROP_THREADS"] == "1"
    assert env["TF_NUM_INTRAOP_THREADS"] == "1"
    assert env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert env["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.8"
    assert env["JAX_PLATFORM_NAME"] == "gpu"
    assert "--xla_gpu_force_compilation_parallelism=1" in env["XLA_FLAGS"]
    assert "--xla_force_host_platform_device_count=1" in env["XLA_FLAGS"]


def test_af2_cluster_wrapper_job_script_exports_gpu_defaults(tmp_path):
    module = _load_module("test_cluster_af2_wrapper_module", AF2_WRAPPER_PATH)
    job = module.JobSpec(
        index=1,
        nodeid="test/cluster/check_alphafold2_predictions.py::TestRunModes::test__monomer",
        slug="af2_node",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        script_path=tmp_path / "job.sh",
        rerun_command="python -m pytest",
    )

    module.write_job_script(
        job=job,
        python_executable=sys.executable,
        use_temp_dir=True,
        cpus_per_task=8,
    )

    script_text = job.script_path.read_text(encoding="utf-8")
    assert 'OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in script_text
    assert 'OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"' in script_text
    assert 'TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"' in script_text
    assert 'JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"' in script_text
    assert "--xla_gpu_force_compilation_parallelism=1" in script_text
    assert "--xla_force_host_platform_device_count=1" in script_text
    assert "addopts=-ra --strict-markers" in script_text
    assert "--use-temp-dir" in script_text


def test_af3_cluster_wrapper_job_script_sets_perf_flag(tmp_path):
    module = _load_module("test_cluster_af3_wrapper_module", AF3_WRAPPER_PATH)
    job = module.JobSpec(
        index=1,
        nodeid="test/cluster/check_alphafold3_predictions.py::TestRunModes::test__monomer",
        slug="af3_node",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        script_path=tmp_path / "job.sh",
        rerun_command="python -m pytest",
    )

    module.write_job_script(
        job=job,
        python_executable=sys.executable,
        use_temp_dir=True,
        include_perf=True,
    )

    script_text = job.script_path.read_text(encoding="utf-8")
    assert "export AF3_RUN_PERF_TESTS=1" in script_text
    assert "addopts=-ra --strict-markers" in script_text
    assert "--use-temp-dir" in script_text
