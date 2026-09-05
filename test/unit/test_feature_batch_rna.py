"""RNA support in the local MMseqs2 feature path.

The tests that need AlphaFold 3 say so; the rest - configuration, validation, cache
identity and the exact MMseqs2 command line - run anywhere, because they are what
guards the promise that a protein-only run is untouched by any of this.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import pytest

from alphapulldown.feature_batch import (
    DEFAULT_RNA_E_VALUE,
    DEFAULT_RNA_MAX_SEQUENCES,
    PROTEIN,
    RNA,
    RNA_DATABASE_NAMES,
    DatabaseSpec,
    FeatureBatch,
    FeatureBatchSettings,
    FeatureRequest,
    MsaBatch,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
    feature_requests_from_fastas,
)

try:
    from alphafold3.common import folding_input as _REAL_FOLDING_INPUT
    from alphafold3.cpp import msa_conversion as _REAL_MSA_CONVERSION
except ImportError as exc:  # pragma: no cover - depends on the environment
    _REAL_FOLDING_INPUT = None
    _REAL_MSA_CONVERSION = None
    _AF3_IMPORT_ERROR = str(exc)
else:
    _AF3_IMPORT_ERROR = ""

requires_af3 = pytest.mark.skipif(
    _REAL_FOLDING_INPUT is None,
    reason=f"AlphaFold 3 test dependencies are unavailable: {_AF3_IMPORT_ERROR}",
)


@pytest.fixture
def real_af3_modules(monkeypatch):
    """Insulate these tests from another module's AlphaFold 3 stubs."""
    monkeypatch.setitem(
        sys.modules, "alphafold3.common.folding_input", _REAL_FOLDING_INPUT
    )
    monkeypatch.setitem(
        sys.modules, "alphafold3.cpp.msa_conversion", _REAL_MSA_CONVERSION
    )
    monkeypatch.setattr(
        sys.modules["alphafold3.common"],
        "folding_input",
        _REAL_FOLDING_INPUT,
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["alphafold3.cpp"],
        "msa_conversion",
        _REAL_MSA_CONVERSION,
        raising=False,
    )


# --------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------


PROTEIN_DATABASE_NAMES = ("uniref90", "mgnify", "small_bfd", "uniprot")
# One base per database, so a merged MSA says which database each row came from. The
# nucleotide databases answer in the DNA alphabet, as the AlphaFold 3 RNA reference
# FASTAs may: that is exactly what has to survive the trip to the model as RNA.
_HIT_BASE = {
    "uniref90": "V",
    "mgnify": "R",
    "small_bfd": "N",
    "uniprot": "D",
    "rfam": "T",
    "rnacentral": "C",
    "nt_rna": "G",
}


class FakeMmseqsProcess:
    """Deterministic stand-in for the external MMseqs2 process."""

    def __init__(self, identity: str = "mmseqs-fixture-1") -> None:
        self._identity = identity
        self._queries: dict[Path, list[tuple[str, str]]] = {}
        self.searched: list[tuple[str, str]] = []

    def identity(self) -> str:
        return self._identity

    def search_mode(self) -> str:
        return "gpu"

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None:
        records = []
        description = None
        parts: list[str] = []
        for line in query_fasta.read_text(encoding="utf-8").splitlines():
            if line.startswith(">"):
                if description is not None:
                    records.append((description, "".join(parts)))
                description = line[1:]
                parts = []
            else:
                parts.append(line.strip())
        if description is not None:
            records.append((description, "".join(parts)))
        self._queries[query_db] = records
        Path(f"{query_db}.lookup").write_text(
            "".join(f"{index}\t{name}\t0\n" for index, (name, _) in enumerate(records)),
            encoding="utf-8",
        )

    def search(self, query_db, database, result_db, work_dir, settings) -> None:
        del work_dir, settings
        self.searched.append((database.name, database.molecule_type))
        for _, sequence in self._queries[query_db]:
            assert sequence
        result_db.write_text(database.name, encoding="utf-8")

    def result_to_msa(self, query_db, database, result_db, msa_db) -> None:
        del query_db, result_db
        msa_db.write_text(database.name, encoding="utf-8")

    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        database_name = msa_db.read_text(encoding="utf-8")
        nucleotide = database_name in RNA_DATABASE_NAMES
        for index, (query_id, sequence) in enumerate(self._queries[query_db]):
            stored_query = sequence.replace("U", "T") if nucleotide else sequence
            hit = _HIT_BASE[database_name] + stored_query[1:]
            (output_dir / f"{index}.a3m").write_text(
                f">{query_id}\n{stored_query}\n>{database_name}_hit\n{hit}\n",
                encoding="utf-8",
            )


class ForbiddenMmseqsProcess:
    def identity(self) -> str:
        return "mmseqs-fixture-1"

    def search_mode(self) -> str:
        return "gpu"

    def __getattr__(self, operation):
        raise AssertionError(f"cache hit unexpectedly launched MMseqs2: {operation}")


class PassthroughAf3Pipeline:
    def process(self, fold_input):
        return fold_input


def _spec(root: Path, name: str, molecule_type: str = PROTEIN) -> DatabaseSpec:
    path = root / name
    path.write_text("fixture", encoding="utf-8")
    Path(f"{path}.index").write_text("fixture-index", encoding="utf-8")
    return DatabaseSpec(
        name=name,
        path=path,
        identifier=f"{name}-2026",
        molecule_type=molecule_type,
    )


def _settings(tmp_path: Path, *, rna: bool = True) -> FeatureBatchSettings:
    databases = tmp_path / "databases"
    databases.mkdir(parents=True, exist_ok=True)
    protein = [_spec(databases, name) for name in PROTEIN_DATABASE_NAMES]
    return FeatureBatchSettings(
        output_dir=tmp_path / "features",
        msa_output_dir=tmp_path / "msas",
        temp_dir=tmp_path / "scratch",
        unpaired_databases=tuple(protein[:3]),
        paired_database=protein[3],
        max_sequences_per_batch=8,
        max_residues_per_batch=1_000,
        threads=4,
        rna_databases=(
            tuple(_spec(databases, name, RNA) for name in RNA_DATABASE_NAMES)
            if rna
            else ()
        ),
        max_template_date="2050-01-01",
        template_seqres_database_id="pdb-seqres-2050",
        template_mmcif_database_id="mmcif-2050",
    )


def _msa_settings(settings: FeatureBatchSettings) -> MsaBatchSettings:
    return MsaBatchSettings(
        output_dir=settings.msa_output_dir,
        temp_dir=settings.temp_dir,
        unpaired_databases=settings.unpaired_databases,
        paired_database=settings.paired_database,
        max_sequences_per_batch=settings.max_sequences_per_batch,
        max_residues_per_batch=settings.max_residues_per_batch,
        threads=settings.threads,
        e_value=settings.e_value,
        rna_databases=settings.rna_databases,
        rna_e_value=settings.rna_e_value,
    )


def _mmseqs_binary(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "mmseqs"
    binary.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "${0}.arguments"\n', encoding="utf-8"
    )
    binary.chmod(0o755)
    return binary, Path(f"{binary}.arguments")


# --------------------------------------------------------------------------------
# The databases AlphaFold 3 actually searches for RNA
# --------------------------------------------------------------------------------


def test_rna_databases_are_the_three_alphafold3_searches_in_its_merge_order():
    """AF3 merges Rfam, RNAcentral and NT-RNA into one unpaired RNA MSA."""
    assert RNA_DATABASE_NAMES == ("rfam", "rnacentral", "nt_rna")
    assert set(DEFAULT_RNA_MAX_SEQUENCES) == set(RNA_DATABASE_NAMES)
    assert set(DEFAULT_RNA_MAX_SEQUENCES.values()) == {10_000}
    assert DEFAULT_RNA_E_VALUE == 1e-3
    assert not set(RNA_DATABASE_NAMES) & set(PROTEIN_DATABASE_NAMES)


# --------------------------------------------------------------------------------
# The protein path is untouched
# --------------------------------------------------------------------------------


def test_protein_cache_signature_is_exactly_what_it_was_before_rna_existed(tmp_path):
    """Pinned literally: changing it silently invalidates every cached protein MSA."""
    batch = MsaBatch(
        settings=_msa_settings(_settings(tmp_path)),
        mmseqs_process=FakeMmseqsProcess(),
    )

    assert batch._cache_signature(PROTEIN) == {
        "schema_version": 4,
        "mmseqs_identity": "mmseqs-fixture-1",
        "search_mode": "gpu",
        "e_value": 1e-4,
        "unpaired_databases": [
            {
                "name": "uniref90",
                "identifier": "uniref90-2026",
                "max_sequences": 10_000,
                "index_size": 13,
            },
            {
                "name": "mgnify",
                "identifier": "mgnify-2026",
                "max_sequences": 5_000,
                "index_size": 13,
            },
            {
                "name": "small_bfd",
                "identifier": "small_bfd-2026",
                "max_sequences": 5_000,
                "index_size": 13,
            },
        ],
        "paired_database": {
            "name": "uniprot",
            "identifier": "uniprot-2026",
            "max_sequences": 50_000,
            "index_size": 13,
        },
    }


def test_configuring_rna_does_not_change_the_protein_cache_signature(tmp_path):
    without_rna = MsaBatch(
        settings=_msa_settings(_settings(tmp_path / "a", rna=False)),
        mmseqs_process=FakeMmseqsProcess(),
    )
    with_rna = MsaBatch(
        settings=_msa_settings(_settings(tmp_path / "b", rna=True)),
        mmseqs_process=FakeMmseqsProcess(),
    )

    assert without_rna._cache_signature(PROTEIN) == with_rna._cache_signature(PROTEIN)


def test_a_protein_bundle_records_no_molecule_type_so_old_bundles_still_match(tmp_path):
    batch = MsaBatch(
        settings=_msa_settings(_settings(tmp_path)),
        mmseqs_process=FakeMmseqsProcess(),
    )

    protein = batch._msa_payload(
        FeatureRequest(name="alpha", sequence="ACDE"), ">query\nACDE\n", ">query\nACDE\n"
    )
    rna = batch._msa_payload(
        FeatureRequest(name="beta", sequence="ACGU", molecule_type=RNA),
        ">query\nACGU\n",
        "",
    )

    assert "moleculeType" not in protein
    assert list(protein) == [
        "schemaVersion",
        "name",
        "sequence",
        "unpairedMsa",
        "pairedMsa",
        "unpairedDepth",
        "pairedDepth",
        "provenance",
    ]
    assert rna["moleculeType"] == RNA


def test_protein_search_command_is_unchanged_by_rna_support(tmp_path):
    binary, arguments = _mmseqs_binary(tmp_path)
    database = DatabaseSpec(
        name="uniref90", path=tmp_path / "uniref90", identifier="fixture"
    )

    SubprocessMmseqsProcess(binary).search(
        tmp_path / "query",
        database,
        tmp_path / "result",
        tmp_path / "work",
        _settings(tmp_path),
    )

    command = arguments.read_text(encoding="utf-8").splitlines()
    assert "--search-type" not in command
    assert command[command.index("--gpu") + 1] == "1"
    assert command[command.index("-e") + 1] == "0.0001"


# --------------------------------------------------------------------------------
# What a nucleotide search asks MMseqs2 for
# --------------------------------------------------------------------------------


def test_nucleotide_search_runs_on_cpu_with_the_nucleotide_scoring_model(tmp_path):
    """MMseqs2's GPU prefilter needs padded protein databases, so RNA stays on CPU."""
    binary, arguments = _mmseqs_binary(tmp_path)
    database = DatabaseSpec(
        name="rfam",
        path=tmp_path / "rfam",
        identifier="fixture",
        molecule_type=RNA,
    )

    SubprocessMmseqsProcess(binary, gpu=True).search(
        tmp_path / "query",
        database,
        tmp_path / "result",
        tmp_path / "work",
        _settings(tmp_path),
    )

    command = arguments.read_text(encoding="utf-8").splitlines()
    assert command[command.index("--search-type") + 1] == "3"
    assert command[command.index("--gpu") + 1] == "0"
    assert command[command.index("-e") + 1] == "0.001"
    assert command[command.index("--max-seqs") + 1] == "10000"


def test_rna_cache_identity_is_separate_from_the_protein_one(tmp_path):
    batch = MsaBatch(
        settings=_msa_settings(_settings(tmp_path)),
        mmseqs_process=FakeMmseqsProcess(),
    )

    signature = batch._cache_signature(RNA)

    assert signature["molecule_type"] == RNA
    assert signature["search_mode"] == "cpu"
    assert signature["e_value"] == DEFAULT_RNA_E_VALUE
    assert [database["name"] for database in signature["unpaired_databases"]] == list(
        RNA_DATABASE_NAMES
    )
    # AlphaFold 3 never pairs RNA chains, so there is nothing paired to record.
    assert "paired_database" not in signature


# --------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------


def test_an_rna_request_needs_all_three_rna_databases(tmp_path):
    settings = _settings(tmp_path, rna=False)
    batch = MsaBatch(
        settings=_msa_settings(settings), mmseqs_process=ForbiddenMmseqsProcess()
    )

    with pytest.raises(ValueError, match="rfam, rnacentral, nt_rna"):
        batch.generate([FeatureRequest(name="r1", sequence="ACGU", molecule_type=RNA)])


def test_a_protein_request_needs_no_rna_databases(tmp_path):
    settings = _msa_settings(_settings(tmp_path, rna=False))

    MsaBatch(settings=settings, mmseqs_process=FakeMmseqsProcess())._validate(
        [FeatureRequest(name="alpha", sequence="ACDE")]
    )


def test_an_rna_only_batch_needs_no_protein_databases(tmp_path):
    """Folding an RNA should not require several hundred gigabytes of protein FASTA."""
    settings = dataclasses.replace(
        _msa_settings(_settings(tmp_path)),
        unpaired_databases=(),
        paired_database=None,
    )

    MsaBatch(settings=settings, mmseqs_process=FakeMmseqsProcess())._validate(
        [FeatureRequest(name="r1", sequence="ACGU", molecule_type=RNA)]
    )


def test_rna_sequences_must_be_written_in_the_rna_alphabet(tmp_path):
    """A T would reach AlphaFold 3 as an unknown nucleotide, not as thymine."""
    batch = MsaBatch(
        settings=_msa_settings(_settings(tmp_path)),
        mmseqs_process=ForbiddenMmseqsProcess(),
    )

    with pytest.raises(ValueError, match="is not a rna sequence"):
        batch.generate([FeatureRequest(name="r1", sequence="ACGT", molecule_type=RNA)])


def test_an_unknown_molecule_type_is_rejected_before_any_search(tmp_path):
    batch = MsaBatch(
        settings=_msa_settings(_settings(tmp_path)),
        mmseqs_process=ForbiddenMmseqsProcess(),
    )

    with pytest.raises(ValueError, match="unsupported molecule type"):
        batch.generate([FeatureRequest(name="d1", sequence="ACGT", molecule_type="dna")])


# --------------------------------------------------------------------------------
# Reading FASTAs
# --------------------------------------------------------------------------------


def test_rna_in_a_fasta_is_rejected_until_the_rna_databases_are_configured(tmp_path):
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna_chain\nACGUACGU\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mmseqs_rfam_database_path"):
        feature_requests_from_fastas([fasta])


def test_rna_and_protein_are_read_from_one_fasta_with_their_types(tmp_path):
    fasta = tmp_path / "mixed.fasta"
    fasta.write_text(">p1\nMKVLA\n>r1\nACGUACGU\n", encoding="utf-8")

    requests = feature_requests_from_fastas([fasta], molecule_types=(PROTEIN, RNA))

    assert [(item.name, item.molecule_type) for item in requests] == [
        ("p1", PROTEIN),
        ("r1", RNA),
    ]


def test_dna_stays_unsupported_even_once_rna_is_configured(tmp_path):
    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">DNA example\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="neither of those"):
        feature_requests_from_fastas([fasta], molecule_types=(PROTEIN, RNA))


# --------------------------------------------------------------------------------
# Flag surface
# --------------------------------------------------------------------------------


def test_rna_database_flags_exist_and_default_to_unset():
    from absl import flags

    from alphapulldown.scripts._mmseqs2_cli import define_msa_search_flags

    define_msa_search_flags()

    for name in RNA_DATABASE_NAMES:
        assert flags.FLAGS[f"mmseqs_{name}_database_path"].default is None
        assert flags.FLAGS[f"mmseqs_{name}_database_id"].default is None
        assert flags.FLAGS[f"mmseqs_{name}_max_sequences"].default == 10_000
    assert flags.FLAGS["mmseqs_rna_e_value"].default == DEFAULT_RNA_E_VALUE


class _Flag:
    def __init__(self, value):
        self.value = value


def _flag_dict(*, rna_paths: bool) -> dict:
    from alphapulldown import feature_batch as fb

    values = {}
    for name in fb.DATABASE_NAMES:
        values[f"mmseqs_{name}_database_path"] = _Flag(f"/db/{name}")
        values[f"mmseqs_{name}_database_id"] = _Flag(f"{name}-id")
        values[f"mmseqs_{name}_max_sequences"] = _Flag(None)
    for name in RNA_DATABASE_NAMES:
        values[f"mmseqs_{name}_database_path"] = _Flag(f"/db/{name}" if rna_paths else None)
        values[f"mmseqs_{name}_database_id"] = _Flag(f"{name}-id")
        values[f"mmseqs_{name}_max_sequences"] = _Flag(None)
    return values


def test_rna_is_enabled_by_configuring_its_databases_and_off_otherwise():
    from alphapulldown.scripts._mmseqs2_cli import (
        accepted_molecule_types,
        database_selection,
    )

    off = database_selection(_flag_dict(rna_paths=False))
    on = database_selection(_flag_dict(rna_paths=True))

    assert off.rna == ()
    assert accepted_molecule_types(_flag_dict(rna_paths=False)) == (PROTEIN,)
    assert [database.name for database in on.rna] == list(RNA_DATABASE_NAMES)
    assert {database.molecule_type for database in on.rna} == {RNA}
    assert accepted_molecule_types(_flag_dict(rna_paths=True)) == (PROTEIN, RNA)
    # The protein roles are assigned exactly as before either way.
    assert [database.name for database in on.unpaired] == [
        "uniref90",
        "mgnify",
        "small_bfd",
    ]
    assert on.paired.name == "uniprot"


def test_configuring_only_some_rna_databases_is_an_error_not_a_partial_opt_in():
    from alphapulldown.scripts._mmseqs2_cli import database_selection

    values = _flag_dict(rna_paths=True)
    values["mmseqs_nt_rna_database_path"] = _Flag(None)

    with pytest.raises(ValueError, match="--mmseqs_nt_rna_database_path"):
        database_selection(values)


def test_required_flags_grow_only_for_the_molecule_types_present():
    from alphapulldown.scripts._mmseqs2_cli import required_msa_flag_names

    protein_only = required_msa_flag_names()
    with_rna = required_msa_flag_names((PROTEIN, RNA))

    assert "mmseqs_uniref90_database_path" in protein_only
    assert not any("rfam" in name for name in protein_only)
    assert set(protein_only) < set(with_rna)
    assert "mmseqs_rfam_database_id" in with_rna
    rna_only = required_msa_flag_names((RNA,))
    assert not any("uniref90" in name for name in rna_only)


def test_missing_database_flags_are_reported_for_the_types_actually_present():
    from absl import flags

    from alphapulldown.scripts._mmseqs2_cli import require_msa_flags

    values = _flag_dict(rna_paths=False)
    values.update(
        {
            name: _Flag("set")
            for name in (
                "msa_output_dir",
                "mmseqs_binary_path",
                "mmseqs_temp_dir",
                "mmseqs_batch_max_sequences",
                "mmseqs_batch_max_residues",
            )
        }
    )

    require_msa_flags(values, {PROTEIN})
    with pytest.raises(
        flags.IllegalFlagValueError, match="--mmseqs_rfam_database_path"
    ):
        require_msa_flags(values, {PROTEIN, RNA})


# --------------------------------------------------------------------------------
# End to end, against the real AlphaFold 3
# --------------------------------------------------------------------------------


@requires_af3
def test_an_rna_only_fold_produces_an_af3_rna_chain_with_one_unpaired_msa(
    tmp_path, real_af3_modules
):
    process = FakeMmseqsProcess()
    result = FeatureBatch(
        settings=_settings(tmp_path),
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([FeatureRequest(name="r1", sequence="ACGUACGU", molecule_type=RNA)])

    assert result.failures == ()
    assert [artifact.name for artifact in result.written] == ["r1"]
    assert process.searched == [(name, RNA) for name in RNA_DATABASE_NAMES]

    payload = json.loads(
        (tmp_path / "features" / "r1_af3_input.json").read_text(encoding="utf-8")
    )
    chain = payload["sequences"][0]["rna"]
    assert chain["sequence"] == "ACGUACGU"
    # One merged unpaired MSA, no paired MSA and no templates: that is all an AF3 RNA
    # chain has.
    assert "pairedMsa" not in chain
    assert "templates" not in chain
    assert chain["unpairedMsa"].count(">query") == 1
    for name in RNA_DATABASE_NAMES:
        assert f">{name}_hit" in chain["unpairedMsa"]


@requires_af3
def test_hits_from_a_dna_alphabet_database_reach_alphafold3_as_rna(
    tmp_path, real_af3_modules
):
    """A T in an MSA row is an unknown nucleotide to AF3, not a base pair."""
    settings = _msa_settings(_settings(tmp_path))
    MsaBatch(settings=settings, mmseqs_process=FakeMmseqsProcess()).generate(
        [FeatureRequest(name="r1", sequence="ACGUACGU", molecule_type=RNA)]
    )

    payload = json.loads(
        (settings.output_dir / "r1_mmseqs_msa.json").read_text(encoding="utf-8")
    )
    assert "T" not in payload["unpairedMsa"]
    assert payload["unpairedMsa"].startswith(">query\nACGUACGU\n")
    # The Rfam fixture answers with a hit whose first base is T, i.e. uracil written
    # in the DNA alphabet; it has to arrive as U.
    assert ">rfam_hit\nUCGUACGU\n" in payload["unpairedMsa"]
    assert payload["pairedMsa"] == ""
    assert payload["unpairedDepth"] == 1 + len(RNA_DATABASE_NAMES)


@requires_af3
def test_a_mixed_fold_sends_each_chain_to_the_databases_for_its_own_type(
    tmp_path, real_af3_modules
):
    process = FakeMmseqsProcess()
    result = FeatureBatch(
        settings=_settings(tmp_path),
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate(
        [
            FeatureRequest(name="p1", sequence="MKVLA"),
            FeatureRequest(name="r1", sequence="ACGUACGU", molecule_type=RNA),
        ]
    )

    assert result.failures == ()
    assert sorted(artifact.name for artifact in result.written) == ["p1", "r1"]
    # Proteins first, then RNA; neither chain is searched against the other's databases.
    assert process.searched == [
        (name, PROTEIN) for name in PROTEIN_DATABASE_NAMES
    ] + [(name, RNA) for name in RNA_DATABASE_NAMES]

    protein = json.loads(
        (tmp_path / "features" / "p1_af3_input.json").read_text(encoding="utf-8")
    )["sequences"][0]["protein"]
    rna = json.loads(
        (tmp_path / "features" / "r1_af3_input.json").read_text(encoding="utf-8")
    )["sequences"][0]["rna"]
    assert ">uniprot_hit" in protein["pairedMsa"]
    assert ">rfam_hit" in rna["unpairedMsa"]
    assert "uniref90" not in rna["unpairedMsa"]


@requires_af3
def test_a_protein_only_batch_never_touches_the_rna_databases(
    tmp_path, real_af3_modules
):
    process = FakeMmseqsProcess()
    FeatureBatch(
        settings=_settings(tmp_path),
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([FeatureRequest(name="p1", sequence="MKVLA")])

    assert process.searched == [(name, PROTEIN) for name in PROTEIN_DATABASE_NAMES]


@requires_af3
def test_an_rna_bundle_is_reused_without_a_second_search(tmp_path, real_af3_modules):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="r1", sequence="ACGUACGU", molecule_type=RNA)
    FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    second = FeatureBatch(
        settings=settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert second.failures == ()
    assert [artifact.name for artifact in second.reused] == ["r1"]


@requires_af3
def test_the_same_letters_as_protein_and_as_rna_are_two_different_searches(
    tmp_path, real_af3_modules
):
    """'ACG' is a legal protein and a legal RNA; one bundle must not answer for both."""
    settings = _msa_settings(_settings(tmp_path))
    process = FakeMmseqsProcess()

    result = MsaBatch(settings=settings, mmseqs_process=process).generate(
        [
            FeatureRequest(name="as_protein", sequence="ACG"),
            FeatureRequest(name="as_rna", sequence="ACG", molecule_type=RNA),
        ]
    )

    assert result.failures == ()
    protein = json.loads(
        (settings.output_dir / "as_protein_mmseqs_msa.json").read_text(encoding="utf-8")
    )
    rna = json.loads(
        (settings.output_dir / "as_rna_mmseqs_msa.json").read_text(encoding="utf-8")
    )
    assert protein["provenance"] != rna["provenance"]
    assert ">uniref90_hit" in protein["unpairedMsa"]
    assert ">rfam_hit" in rna["unpairedMsa"]
