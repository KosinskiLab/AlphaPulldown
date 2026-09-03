"""Build contracts for the MMseqs2-GPU runtime shared by both images."""

from pathlib import Path
import re

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
DOCKERFILES = (
    REPOSITORY / "docker" / "alphafold2.dockerfile",
    REPOSITORY / "docker" / "alphafold3.dockerfile",
)
EXPECTED_VERSION = "18-8cc5c"
EXPECTED_SHA256 = "83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7"


def _build_argument(source: str, name: str) -> str:
    match = re.search(rf"^ARG {name}=([^\s]+)$", source, flags=re.MULTILINE)
    assert match is not None, f"missing {name} build pin"
    return match.group(1)


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_prediction_images_install_the_same_verified_mmseqs2_gpu_release(dockerfile):
    source = dockerfile.read_text(encoding="utf-8")

    assert _build_argument(source, "MMSEQS_VERSION") == EXPECTED_VERSION
    assert _build_argument(source, "MMSEQS_GPU_SHA256") == EXPECTED_SHA256
    assert (
        "https://github.com/soedinglab/MMseqs2/releases/download/"
        "${MMSEQS_VERSION}/mmseqs-linux-gpu.tar.gz"
    ) in source
    assert "sha256sum -c -" in source
    assert "/opt/mmseqs/bin/mmseqs version" in source
