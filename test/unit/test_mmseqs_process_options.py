"""Process-level MMseqs2 options, and which of them may touch the cache identity.

Deliberately free of any AlphaFold 3 import: the search stage runs in the
AlphaFold 2 image too, and a module-level skip there would report success while
silently testing nothing. The equivalent assertions in ``test_feature_batch.py``
cannot run in that image because that module needs AlphaFold 3 for the finalizer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from alphapulldown.feature_batch import (
    DatabaseSpec,
    MsaBatch,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
)


PROTEIN_DATABASES = ("uniref90", "mgnify", "small_bfd")


@pytest.fixture
def recording_binary(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in executable that records the arguments it was called with."""
    binary = tmp_path / "mmseqs"
    binary.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "${0}.arguments"\n', encoding="utf-8"
    )
    binary.chmod(0o755)
    return binary, Path(f"{binary}.arguments")


def _settings(tmp_path: Path) -> MsaBatchSettings:
    return MsaBatchSettings(
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "tmp",
        unpaired_databases=tuple(
            DatabaseSpec(name=name, path=tmp_path / name, identifier=f"{name}-fixture")
            for name in PROTEIN_DATABASES
        ),
        paired_database=DatabaseSpec(
            name="uniprot", path=tmp_path / "uniprot", identifier="uniprot-fixture"
        ),
        max_sequences_per_batch=8,
        max_residues_per_batch=10_000,
        threads=4,
    )


def _database(tmp_path: Path) -> DatabaseSpec:
    return DatabaseSpec(
        name="uniref90", path=tmp_path / "uniref90", identifier="fixture"
    )


def test_db_load_mode_reaches_the_search(tmp_path: Path, recording_binary):
    binary, arguments = recording_binary
    SubprocessMmseqsProcess(binary, db_load_mode=2).search(
        tmp_path / "query",
        _database(tmp_path),
        tmp_path / "result",
        tmp_path / "work",
        _settings(tmp_path),
    )
    command = arguments.read_text(encoding="utf-8").splitlines()
    assert command[command.index("--db-load-mode") + 1] == "2"


def test_db_load_mode_reaches_result_to_msa(tmp_path: Path, recording_binary):
    # Both commands read the target database, so both must honour the setting;
    # passing it only to the search would leave the peak memory it exists to
    # lower untouched in the second half of the stage.
    binary, arguments = recording_binary
    SubprocessMmseqsProcess(binary, db_load_mode=2).result_to_msa(
        tmp_path / "query",
        _database(tmp_path),
        tmp_path / "result",
        tmp_path / "msa",
    )
    command = arguments.read_text(encoding="utf-8").splitlines()
    assert command[command.index("--db-load-mode") + 1] == "2"


@pytest.mark.parametrize("operation", ("search", "result_to_msa"))
def test_the_option_is_absent_when_unset(
    tmp_path: Path, recording_binary, operation: str
):
    """Unset must mean "say nothing", so MMseqs2 keeps choosing for itself."""
    binary, arguments = recording_binary
    process = SubprocessMmseqsProcess(binary)
    if operation == "search":
        process.search(
            tmp_path / "query",
            _database(tmp_path),
            tmp_path / "result",
            tmp_path / "work",
            _settings(tmp_path),
        )
    else:
        process.result_to_msa(
            tmp_path / "query",
            _database(tmp_path),
            tmp_path / "result",
            tmp_path / "msa",
        )
    assert "--db-load-mode" not in arguments.read_text().splitlines()


def test_db_load_mode_stays_out_of_the_cache_signature(tmp_path: Path):
    """It changes memory behaviour, never the alignment.

    If it reached the signature, turning it on to survive a tight allocation
    would discard every alignment already computed -- a re-search costing hours
    per shard to produce byte-identical output. Asserted on two real adapters
    that differ in exactly this one setting.
    """
    binary = tmp_path / "mmseqs"
    binary.write_text("#!/bin/sh\necho mmseqs-fixture-version\n", encoding="utf-8")
    binary.chmod(0o755)
    settings = _settings(tmp_path)

    plain = MsaBatch(
        settings=settings, mmseqs_process=SubprocessMmseqsProcess(binary)
    )
    mapped = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(binary, db_load_mode=2),
    )
    assert plain._cache_signature() == mapped._cache_signature()

    # The comparison above is only meaningful if a setting that DOES change the
    # alignment moves the signature. GPU and CPU search are such a setting.
    on_cpu = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(binary, gpu=False),
    )
    assert plain._cache_signature() != on_cpu._cache_signature()


def test_iterations_reach_protein_searches_but_never_nucleotide_ones(
    tmp_path: Path, recording_binary
):
    """Iterative profile search is protein-only. Measured on the pinned build,
    --num-iterations 3 on a nucleotide search exits 1, so raising the setting
    used to fail every RNA shard."""
    binary, arguments = recording_binary
    settings = dataclasses.replace(_settings(tmp_path), num_iterations=3)
    process = SubprocessMmseqsProcess(binary)

    process.search(
        tmp_path / "query",
        _database(tmp_path),
        tmp_path / "result",
        tmp_path / "work",
        settings,
    )
    protein = arguments.read_text(encoding="utf-8").splitlines()
    assert protein[protein.index("--num-iterations") + 1] == "3"

    rfam = DatabaseSpec(
        name="rfam", path=tmp_path / "rfam", identifier="rfam-fixture",
        molecule_type="rna",
    )
    process.search(
        tmp_path / "query", rfam, tmp_path / "result", tmp_path / "work", settings
    )
    nucleotide = arguments.read_text(encoding="utf-8").splitlines()
    assert "--num-iterations" not in nucleotide
    assert nucleotide[nucleotide.index("--search-type") + 1] == "3"
