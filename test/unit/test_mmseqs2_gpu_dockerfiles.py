"""Build contracts for the MMseqs2-GPU runtime shared by both images."""

from pathlib import Path
import re

import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
DOCKERFILES = (
    REPOSITORY / "docker" / "alphafold2.dockerfile",
    REPOSITORY / "docker" / "alphafold3.dockerfile",
)
EXPECTED_VERSION = "18-8cc5c"
EXPECTED_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
EXPECTED_SHA256 = "83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7"
WORKFLOW = REPOSITORY / ".github" / "workflows" / "github_actions.yml"


def _build_argument(source: str, name: str) -> str:
    match = re.search(rf"^ARG {name}=([^\s]+)$", source, flags=re.MULTILINE)
    assert match is not None, f"missing {name} build pin"
    return match.group(1)


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_prediction_images_install_the_same_verified_mmseqs2_gpu_release(dockerfile):
    source = dockerfile.read_text(encoding="utf-8")

    assert _build_argument(source, "MMSEQS_VERSION") == EXPECTED_VERSION
    assert _build_argument(source, "MMSEQS_COMMIT") == EXPECTED_COMMIT
    assert _build_argument(source, "MMSEQS_GPU_SHA256") == EXPECTED_SHA256
    assert (
        "https://github.com/soedinglab/MMseqs2/releases/download/"
        "${MMSEQS_VERSION}/mmseqs-linux-gpu.tar.gz"
    ) in source
    assert "sha256sum -c -" in source
    assert 'test "$(/opt/mmseqs/bin/mmseqs version)" = "${MMSEQS_COMMIT}"' in source


def test_alphafold2_container_builds_on_pull_requests_without_secrets():
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["build-alphafold2-container"]["steps"]

    ssh_agent = next(
        step
        for step in steps
        if step.get("uses", "").startswith("webfactory/ssh-agent@")
    )
    registry_login = next(
        step
        for step in steps
        if step.get("uses", "").startswith("docker/login-action@")
    )
    pull_request_build = next(
        step
        for step in steps
        if step.get("name", "").startswith("Build alphafold2 container")
        and "push" not in step.get("name", "")
    )

    assert ssh_agent["if"] == "github.event_name != 'pull_request'"
    assert registry_login["if"] == "github.event_name != 'pull_request'"
    assert pull_request_build["if"] == "github.event_name == 'pull_request'"
    assert pull_request_build["uses"].startswith("docker/build-push-action@")
    assert pull_request_build["with"]["context"] == "."
    assert pull_request_build["with"]["file"] == "./docker/alphafold2.dockerfile"
    assert pull_request_build["with"]["push"] == "false"
    assert "ssh" not in pull_request_build["with"]

    publish_steps = [
        step for step in steps if step.get("with", {}).get("push") == "true"
    ]
    assert {step["if"] for step in publish_steps} == {
        "github.event_name == 'push'",
        "github.event_name == 'release' && github.event.action == 'published'",
    }
    assert all(step["with"]["ssh"] == "default" for step in publish_steps)


def test_alphafold3_pr_build_runs_compiled_feature_batch_contracts():
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["build-alphafold3-container"]["steps"]
    pull_request_build = next(
        step
        for step in steps
        if step.get("name")
        == "Build alphafold3 container and run compiled-AF3 compatibility tests"
    )
    assert pull_request_build["if"] == "github.event_name == 'pull_request'"
    assert pull_request_build["with"]["context"] == "."
    assert pull_request_build["with"]["file"] == "./docker/alphafold3.dockerfile"
    assert pull_request_build["with"]["push"] == "false"

    dockerfile = (REPOSITORY / "docker" / "alphafold3.dockerfile").read_text(
        encoding="utf-8"
    )
    assert "test/unit/test_feature_batch.py" in dockerfile
    assert "test/unit/test_create_batch_features.py" in dockerfile
    assert "MMSEQS_INTEGRATION_BINARY=/opt/mmseqs/bin/mmseqs" in dockerfile
    assert "test/integration/test_mmseqs2_command_contract.py" in dockerfile
    assert 'addopts="-ra --strict-markers"' in dockerfile


def test_alphafold3_image_tests_the_fused_triangles_and_one_compile_per_process():
    dockerfile = (REPOSITORY / "docker" / "alphafold3.dockerfile").read_text(
        encoding="utf-8"
    )
    stage = dockerfile.split("FROM base AS af3-compatibility-tests", 1)[1].split(
        "FROM base AS runtime", 1
    )[0]

    for test_path in (
        "test/unit/test_af3_fused_triangles.py",
        "test/unit/test_alphafold3_backend_helpers.py",
        "test/unit/test_af3_tokamax_trace_context.py",
        "test/integration/test_af3_input_preparation.py",
        "alphafold3/src/alphafold3/model/network/fused_triangle_test.py",
    ):
        assert test_path in stage, test_path
        assert (REPOSITORY / test_path).is_file(), test_path
    # As a script, so it imports the installed AF3 and its built CCD data.
    assert (
        "JAX_PLATFORMS=cpu python alphafold3/src/alphafold3/model/network/"
        "fused_triangle_test.py"
    ) in stage
    # The runtime image cannot be built without the test stage passing.
    assert "FROM base AS runtime" in dockerfile
    assert "COPY --from=af3-compatibility-tests" in dockerfile.split(
        "FROM base AS runtime", 1
    )[1]


def test_smoke_tests_run_the_alphafold_fork_fused_kernel_hooks():
    # pytest.ini never collects the alphafold/ submodule; CI must name the file.
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["smoke-tests"]["steps"]
    hook_tests = [
        step
        for step in steps
        if "alphafold/alphafold/model/fused_kernels_test.py" in step.get("run", "")
    ]

    assert len(hook_tests) == 1
    assert (
        REPOSITORY / "alphafold" / "alphafold" / "model" / "fused_kernels_test.py"
    ).is_file()
    # The fork needs Python 3.11's typing; the matrix's other entries must run it.
    assert hook_tests[0]["if"] == "matrix.python-version != '3.10'"
    assert "3.11" in workflow["jobs"]["smoke-tests"]["strategy"]["matrix"]["python-version"]


def test_alphafold2_image_checks_the_fused_kernels_import_with_its_final_jax():
    dockerfile = (REPOSITORY / "docker" / "alphafold2.dockerfile").read_text(
        encoding="utf-8"
    )
    kernels = dockerfile.index('"colabfold-kernels>=0.4.0,<0.5"')
    jax = dockerfile.index('"jax[cuda12]"==0.5.3')
    numpy = dockerfile.index('RUN pip install --no-cache-dir "numpy<2"')
    check = dockerfile.index("fused_kernels.fused_ops()")

    # After the last dependency change, so it checks what the image ships.
    assert kernels < jax < numpy < check
    assert "pip install" not in dockerfile[check:]


def test_alphafold2_image_pins_openmm_to_cuda12_for_gpu_relax():
    dockerfile = (REPOSITORY / "docker" / "alphafold2.dockerfile").read_text(
        encoding="utf-8"
    )
    install = dockerfile[dockerfile.index('"openmm>=8.2"'):dockerfile.index("micromamba clean")]

    # Unpinned, conda-forge OpenMM takes its CUDA 13 build, whose NVRTC 13.4 PTX the
    # cluster's CUDA 13.1 drivers cannot load: GPU relax then fails every attempt.
    assert '"cuda-version=12.8"' in install
    assert "ls /opt/conda/conda-meta/cuda-nvrtc-12.*.json" in dockerfile
    assert "! ls /opt/conda/conda-meta/cuda-nvrtc-13.*.json" in dockerfile
    assert dockerfile.count("micromamba install") == 1
