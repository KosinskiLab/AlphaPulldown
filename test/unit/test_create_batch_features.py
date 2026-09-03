import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from alphapulldown.scripts.create_batch_features import FLAGS, _feature_requests


def test_import_forces_jax_to_cpu_without_hiding_gpu_from_mmseqs():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cuda"
    environment["CUDA_VISIBLE_DEVICES"] = "7"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository), environment.get("PYTHONPATH")))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os\n"
                "try:\n"
                "    import alphapulldown.scripts.create_batch_features\n"
                "except ImportError:\n"
                "    pass\n"
                "print(json.dumps({"
                "'jax': os.environ.get('JAX_PLATFORMS'), "
                "'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
            ),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "jax": "cpu",
        "cuda": "7",
    }


def test_cli_adapter_rejects_non_protein_af3_fasta(tmp_path: Path):
    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">DNA example\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protein"):
        _feature_requests([str(fasta)])


def test_cli_defaults_to_the_binary_bundled_in_prediction_images():
    assert FLAGS["mmseqs_binary_path"].default == "/opt/mmseqs/bin/mmseqs"
