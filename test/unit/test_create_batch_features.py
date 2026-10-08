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


def test_msa_stage_imports_neither_jax_nor_alphafold():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
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
                "import json, sys\n"
                "import alphapulldown.scripts.create_batch_msas\n"
                "print(json.dumps({'jax': 'jax' in sys.modules, "
                "'af2': 'alphafold' in sys.modules, "
                "'af3': 'alphafold3' in sys.modules}))\n"
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
        "jax": False,
        "af2": False,
        "af3": False,
    }


def test_lightweight_mmseqs_flag_schema_can_be_registered_twice_without_jax():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
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
                "import importlib.abc, sys\n"
                "class BlockJax(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path, target=None):\n"
                "        if fullname == 'jax' or fullname.startswith('jax.'):\n"
                "            raise ModuleNotFoundError('JAX import forbidden')\n"
                "        return None\n"
                "sys.meta_path.insert(0, BlockJax())\n"
                "import alphapulldown.scripts.create_batch_msas\n"
                "from alphapulldown.scripts._mmseqs2_cli import "
                "define_msa_search_flags, define_template_provenance_flags\n"
                "define_msa_search_flags(include_fasta_paths=True, "
                "include_summary_path=True)\n"
                "define_template_provenance_flags()\n"
                "define_template_provenance_flags()\n"
                "from absl import flags\n"
                "print(flags.FLAGS['mmseqs_uniref90_max_sequences'].default)\n"
            ),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1] == "10000"


def test_cli_adapter_rejects_non_protein_af3_fasta(tmp_path: Path):
    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">DNA example\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protein"):
        _feature_requests([str(fasta)])


def test_cli_adapter_accepts_rna_once_the_rna_databases_are_configured(
    tmp_path: Path, monkeypatch
):
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna_chain\nACGUACGU\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mmseqs_rfam_database_path"):
        _feature_requests([str(fasta)])

    for name in ("rfam", "rnacentral", "nt_rna"):
        monkeypatch.setattr(
            FLAGS[f"mmseqs_{name}_database_path"], "value", f"/db/{name}"
        )

    assert [
        (request.name, request.molecule_type)
        for request in _feature_requests([str(fasta)])
    ] == [("rna_chain", "rna")]


def test_cli_defaults_to_the_binary_bundled_in_prediction_images():
    assert FLAGS["mmseqs_binary_path"].default == "/opt/mmseqs/bin/mmseqs"


@pytest.fixture
def batch_cli(monkeypatch, tmp_path: Path):
    """create_batch_features.main with its AF3 and MMseqs2 collaborators faked."""
    from types import SimpleNamespace

    import alphapulldown.scripts.create_batch_features as cli

    if not FLAGS.is_parsed():
        FLAGS.mark_as_parsed()
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">alpha\nACDEFGHIK\n>beta\nLMNPQRST\n", encoding="utf-8")
    for name, value in {
        "data_pipeline": "alphafold3",
        "fasta_paths": [str(fasta)],
        "output_dir": str(tmp_path / "features"),
        "msa_output_dir": str(tmp_path / "msas"),
        "mmseqs_temp_dir": str(tmp_path / "tmp"),
        "max_template_date": "2021-09-30",
        "use_mmseqs2": False,
        "keep_msas": False,
        "skip_msa": False,
        "path_to_mmt": None,
    }.items():
        monkeypatch.setattr(FLAGS[name], "value", value)
    monkeypatch.setattr(
        cli,
        "legacy_features",
        SimpleNamespace(
            af3_pipeline_settings=lambda: SimpleNamespace(
                flag_values_with_resolved_paths=lambda values: values
            ),
            create_pipeline_af3=lambda: "af3-pipeline",
            get_af3_feature_metadata=lambda kinds, skip_msa, flag_values: {"skip_msa": skip_msa},
        ),
    )
    monkeypatch.setattr(cli, "require_msa_flags", lambda flag_values, kinds: None)
    monkeypatch.setattr(
        cli,
        "database_selection",
        lambda flag_values: SimpleNamespace(unpaired=("uniref90",), paired="uniprot", rna=()),
    )
    monkeypatch.setattr(cli, "SubprocessMmseqsProcess", lambda *args, **kwargs: "mmseqs")
    batches = []

    class FakeFeatureBatch:
        result = SimpleNamespace(written=("alpha", "beta"), reused=(), failures=(), query_only=())

        def __init__(self, *, settings, mmseqs_process, af3_pipeline):
            batches.append((settings, mmseqs_process, af3_pipeline))

        def generate(self, requests):
            batches.append([request.name for request in requests])
            return self.result

    monkeypatch.setattr(cli, "FeatureBatch", FakeFeatureBatch)
    return SimpleNamespace(cli=cli, batches=batches, batch=FakeFeatureBatch, tmp_path=tmp_path)


def test_main_runs_one_feature_batch_over_the_fasta(batch_cli):
    batch_cli.cli.main(["prog"])

    (settings, process, pipeline), names = batch_cli.batches
    assert names == ["alpha", "beta"]
    assert (process, pipeline) == ("mmseqs", "af3-pipeline")
    assert settings.output_dir == batch_cli.tmp_path / "features"
    assert settings.unpaired_databases == ("uniref90",)
    assert settings.paired_database == "uniprot"
    assert settings.base_metadata == {"skip_msa": True}


def test_main_reports_failed_and_query_only_batches(batch_cli):
    from types import SimpleNamespace

    batch_cli.batch.result = SimpleNamespace(
        written=("alpha",), reused=(), failures=(SimpleNamespace(name="beta", error="boom"),),
        query_only=(),
    )
    with pytest.raises(RuntimeError, match=r"Failed to create 1 artifact\(s\): beta \(boom\)"):
        batch_cli.cli.main(["prog"])

    batch_cli.batch.result = SimpleNamespace(
        written=("alpha",), reused=("beta",), failures=(), query_only=("alpha", "beta"),
    )
    with pytest.raises(RuntimeError, match="only the query sequence"):
        batch_cli.cli.main(["prog"])


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("data_pipeline", "alphafold2", "require --data_pipeline=alphafold3"),
        ("use_mmseqs2", True, "cannot be combined"),
        ("path_to_mmt", "/templates", "cannot be combined"),
    ],
)
def test_main_refuses_settings_the_batch_mode_cannot_honour(
    batch_cli, monkeypatch, name, value, message
):
    monkeypatch.setattr(FLAGS[name], "value", value)

    with pytest.raises(ValueError, match=message):
        batch_cli.cli.main(["prog"])
    assert batch_cli.batches == []
