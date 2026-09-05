"""Opt-in contract test against a real MMseqs2 executable and tiny database."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external_tools]

try:
    from alphafold3.cpp import msa_conversion as _msa_conversion  # noqa: F401
except ImportError as exc:
    pytest.skip(
        f"AlphaFold 3 MSA conversion is unavailable: {exc}", allow_module_level=True
    )

from alphapulldown.feature_batch import (  # noqa: E402
    DatabaseSpec,
    FeatureRequest,
    MsaBatch,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
)


def _run(binary: Path, *arguments: str) -> None:
    subprocess.run(
        [str(binary), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_createdb_padded_search_result2msa_and_unpack_contract(tmp_path):
    configured_binary = os.environ.get("MMSEQS_INTEGRATION_BINARY")
    if not configured_binary:
        pytest.skip("set MMSEQS_INTEGRATION_BINARY to run the real command contract")
    binary = Path(configured_binary)
    if not binary.is_file():
        pytest.fail(f"MMSEQS_INTEGRATION_BINARY does not exist: {binary}")

    query_sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQYQKVEKLLKQGADVVVT"
    target_sequence = query_sequence[:-1] + "A"
    target_fasta = tmp_path / "target.fasta"
    target_fasta.write_text(
        f">target_hit expected description OX=9606\n{target_sequence}\n",
        encoding="utf-8",
    )
    target_db = tmp_path / "target"
    padded_db = tmp_path / "target_gpu"
    _run(binary, "createdb", str(target_fasta), str(target_db), "--threads", "1")
    _run(binary, "makepaddedseqdb", str(target_db), str(padded_db), "--threads", "1")

    databases = tuple(
        DatabaseSpec(name=name, path=padded_db, identifier="tiny-padded-v1")
        for name in ("uniref90", "mgnify", "small_bfd", "uniprot")
    )
    settings = MsaBatchSettings(
        output_dir=tmp_path / "msas",
        temp_dir=tmp_path / "work",
        unpaired_databases=databases[:3],
        paired_database=databases[3],
        max_sequences_per_batch=8,
        max_residues_per_batch=1_000,
        threads=2,
    )

    use_gpu = os.environ.get("MMSEQS_INTEGRATION_GPU") == "1"
    result = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(binary, gpu=use_gpu),
    ).generate([FeatureRequest(name="query", sequence=query_sequence)])

    assert result.failures == ()
    payload = json.loads(
        (settings.output_dir / "query_mmseqs_msa.json").read_text(encoding="utf-8")
    )
    assert "target_hit expected description OX=9606" in payload["unpairedMsa"]
    assert "target_hit expected description OX=9606" in payload["pairedMsa"]
    assert payload["provenance"]["search_mode"] == (
        "gpu" if use_gpu else "cpu"
    )
