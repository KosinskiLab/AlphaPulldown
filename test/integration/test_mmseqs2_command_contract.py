"""Opt-in contract test against a real MMseqs2 executable and tiny database."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external_tools]

# No AlphaFold 3 skip here any more. This module exercises the shared protein
# search, which the AlphaFold 2 image has to run too -- and that image has no
# AlphaFold 3 at all. Skipping the whole module on its absence meant an AlphaFold 2
# environment reported success while silently skipping the one test that runs a
# real MMseqs2 search end to end.
from alphapulldown.feature_batch import (
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
    # A UniProt-headed hit with a four-residue insertion. result2msa mode 2 keeps
    # this header and drops the insertion; mode 5 keeps the insertion and cuts the
    # header to "P0CTR1". The bundle has to carry both, on the same row.
    inserted_sequence = query_sequence[:20] + "WWWW" + query_sequence[20:]
    target_fasta = tmp_path / "target.fasta"
    target_fasta.write_text(
        f">target_hit expected description OX=9606\n{target_sequence}\n"
        f">sp|P0CTR1|INSRT_HUMAN inserted OS=Homo sapiens OX=9606\n"
        f"{inserted_sequence}\n",
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

    # The point of the two-pass format, against the real binary: the species
    # header and the insertion arrive together, on one row, in the paired MSA
    # AlphaFold pairs chains from.
    paired = payload["pairedMsa"].splitlines()
    header_index = next(
        index for index, line in enumerate(paired) if line.startswith(">sp|P0CTR1|")
    )
    assert paired[header_index] == (
        ">sp|P0CTR1|INSRT_HUMAN inserted OS=Homo sapiens OX=9606"
    )
    assert "wwww" in paired[header_index + 1], paired[header_index + 1]
    assert (
        paired[header_index + 1].replace("wwww", "").upper() == query_sequence
    ), "removing the insertion must leave the row aligned to the query"

    # And the bundle says which database contributed which unpaired rows.
    assert [span["name"] for span in payload["unpairedDatabaseRows"]] == [
        "uniref90",
        "mgnify",
        "small_bfd",
    ]
    assert (
        sum(span["rows"] for span in payload["unpairedDatabaseRows"])
        == payload["unpairedDepth"] - 1
    )


def test_real_nucleotide_createdb_search_and_unpack_contract(tmp_path):
    """The same contract for RNA: a nucleotide database, searched on CPU."""
    configured_binary = os.environ.get("MMSEQS_INTEGRATION_BINARY")
    if not configured_binary:
        pytest.skip("set MMSEQS_INTEGRATION_BINARY to run the real command contract")
    binary = Path(configured_binary)
    if not binary.is_file():
        pytest.fail(f"MMSEQS_INTEGRATION_BINARY does not exist: {binary}")

    query_sequence = (
        "GGCUAUAGCUCAGUUGGUUAGAGCACAUCACUCAUAAUGAUGGGGUCACAGGUUCGAAUCCUGUUAGCCUAA"
    )
    # A hit written in the DNA alphabet, as an RNA reference FASTA may well be.
    target_sequence = query_sequence.replace("U", "T")[:-1] + "G"
    target_fasta = tmp_path / "target.fasta"
    target_fasta.write_text(
        f">rna_target_hit tRNA-like\n{target_sequence}\n", encoding="utf-8"
    )
    target_db = tmp_path / "rna_target"
    _run(binary, "createdb", str(target_fasta), str(target_db), "--threads", "1")

    databases = tuple(
        DatabaseSpec(
            name=name,
            path=target_db,
            identifier="tiny-nucleotide-v1",
            molecule_type="rna",
        )
        for name in ("rfam", "rnacentral", "nt_rna")
    )
    settings = MsaBatchSettings(
        output_dir=tmp_path / "msas",
        temp_dir=tmp_path / "work",
        unpaired_databases=(),
        paired_database=None,
        max_sequences_per_batch=8,
        max_residues_per_batch=1_000,
        threads=2,
        rna_databases=databases,
    )

    result = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(binary, gpu=False),
    ).generate(
        [FeatureRequest(name="query", sequence=query_sequence, molecule_type="rna")]
    )

    assert result.failures == ()
    payload = json.loads(
        (settings.output_dir / "query_mmseqs_msa.json").read_text(encoding="utf-8")
    )
    assert payload["moleculeType"] == "rna"
    # RNA chains have no paired MSA, and nothing reaches AlphaFold 3 spelled with T.
    assert payload["pairedMsa"] == ""
    assert "T" not in payload["unpairedMsa"]
    assert "rna_target_hit tRNA-like" in payload["unpairedMsa"]
    assert payload["provenance"]["search_mode"] == "cpu"
