from __future__ import annotations

import dataclasses
import json
import lzma
from pathlib import Path
import sys

import pytest

_REAL_FOLDING_INPUT = pytest.importorskip("alphafold3.common.folding_input")
_REAL_AF3_PIPELINE = pytest.importorskip("alphafold3.data.pipeline")

from alphapulldown.feature_batch import (
    DatabaseSpec,
    FeatureBatch,
    FeatureBatchSettings,
    FeatureRequest,
)


@pytest.fixture(autouse=True)
def _use_real_af3_modules(monkeypatch):
    """Insulate these integration tests from another module's AF3 stubs."""
    common_package = sys.modules["alphafold3.common"]
    data_package = sys.modules["alphafold3.data"]
    monkeypatch.setitem(
        sys.modules, "alphafold3.common.folding_input", _REAL_FOLDING_INPUT
    )
    monkeypatch.setitem(sys.modules, "alphafold3.data.pipeline", _REAL_AF3_PIPELINE)
    monkeypatch.setattr(
        common_package, "folding_input", _REAL_FOLDING_INPUT, raising=False
    )
    monkeypatch.setattr(data_package, "pipeline", _REAL_AF3_PIPELINE, raising=False)


class FakeMmseqsProcess:
    """Deterministic stand-in for the external MMseqs2 process."""

    def __init__(self) -> None:
        self._queries: dict[Path, list[tuple[str, str]]] = {}

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None:
        records = []
        description = None
        sequence_parts = []
        for line in query_fasta.read_text(encoding="utf-8").splitlines():
            if line.startswith(">"):
                if description is not None:
                    records.append((description, "".join(sequence_parts)))
                description = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())
        if description is not None:
            records.append((description, "".join(sequence_parts)))

        assert len({sequence for _, sequence in records}) == len(records)
        self._queries[query_db] = records
        Path(f"{query_db}.lookup").write_text(
            "".join(f"{index}\t{name}\t0\n" for index, (name, _) in enumerate(records)),
            encoding="utf-8",
        )

    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: FeatureBatchSettings,
    ) -> None:
        del query_db, work_dir, settings
        result_db.write_text(database.name, encoding="utf-8")

    def result_to_msa(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        msa_db: Path,
    ) -> None:
        del query_db, result_db
        msa_db.write_text(database.name, encoding="utf-8")

    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        database_name = msa_db.read_text(encoding="utf-8")
        for index, (query_id, sequence) in enumerate(self._queries[query_db]):
            (output_dir / f"{index}.a3m").write_text(
                f">{query_id}\n{sequence}\n>{database_name}_hit\n{sequence}\n",
                encoding="utf-8",
            )


class ForbiddenMmseqsProcess:
    def __getattr__(self, operation):
        raise AssertionError(f"cache hit unexpectedly launched MMseqs2: {operation}")


class FirstSequenceFailsMmseqs(FakeMmseqsProcess):
    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: FeatureBatchSettings,
    ) -> None:
        if self._queries[query_db][0][1] == "ACDE":
            raise RuntimeError("fixture MMseqs2 failure")
        super().search(query_db, database, result_db, work_dir, settings)


class PassthroughAf3Pipeline:
    def process(self, fold_input):
        return fold_input


def _native_pipeline_with_no_template_hits(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hmmbuild = bin_dir / "hmmbuild"
    hmmbuild.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[-2]).write_text('HMM\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    hmmsearch = bin_dir / "hmmsearch"
    hmmsearch.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('-A') + 1]).write_text('', encoding='utf-8')\n",
        encoding="utf-8",
    )
    hmmbuild.chmod(0o755)
    hmmsearch.chmod(0o755)

    sequence_database = tmp_path / "pdb_seqres.fasta"
    sequence_database.write_text(">none\nAAAA\n", encoding="utf-8")
    structure_directory = tmp_path / "mmcif"
    structure_directory.mkdir()

    config = _REAL_AF3_PIPELINE.DataPipelineConfig(
        jackhmmer_binary_path=str(hmmbuild),
        nhmmer_binary_path=str(hmmbuild),
        hmmalign_binary_path=str(hmmbuild),
        hmmsearch_binary_path=str(hmmsearch),
        hmmbuild_binary_path=str(hmmbuild),
        small_bfd_database_path=str(sequence_database),
        mgnify_database_path=str(sequence_database),
        uniprot_cluster_annot_database_path=str(sequence_database),
        uniref90_database_path=str(sequence_database),
        ntrna_database_path=str(sequence_database),
        rfam_database_path=str(sequence_database),
        rna_central_database_path=str(sequence_database),
        seqres_database_path=str(sequence_database),
        pdb_database_path=str(structure_directory),
        max_template_date=__import__("datetime").date(2050, 1, 1),
    )
    return _REAL_AF3_PIPELINE.DataPipeline(config)


def _settings(tmp_path: Path) -> FeatureBatchSettings:
    databases = tmp_path / "databases"
    databases.mkdir()
    specs = []
    for name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
        path = databases / name
        path.write_text("fixture", encoding="utf-8")
        specs.append(DatabaseSpec(name=name, path=path, identifier=f"{name}-2026"))
    return FeatureBatchSettings(
        output_dir=tmp_path / "features",
        temp_dir=tmp_path / "scratch",
        unpaired_databases=tuple(specs[:3]),
        paired_database=specs[3],
        max_sequences_per_batch=8,
        max_residues_per_batch=1_000,
        threads=4,
    )


def test_duplicate_sequences_are_searched_once_and_create_standard_artifacts(tmp_path):
    batch = FeatureBatch(
        settings=_settings(tmp_path),
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=_native_pipeline_with_no_template_hits(tmp_path),
    )

    result = batch.generate(
        [
            FeatureRequest(name="alpha", sequence="ACDEFG"),
            FeatureRequest(name="beta", sequence="ACDEFG"),
        ]
    )

    assert result.failures == ()
    assert [artifact.name for artifact in result.written] == ["alpha", "beta"]
    for name in ("alpha", "beta"):
        payload = json.loads(
            (tmp_path / "features" / f"{name}_af3_input.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["name"] == name
        protein = payload["sequences"][0]["protein"]
        assert protein["sequence"] == "ACDEFG"
        assert protein["unpairedMsa"].count(">query") == 1
        assert ">uniprot_hit" in protein["pairedMsa"]
        assert protein["templates"] == []


def test_matching_artifact_is_reused_without_external_search(tmp_path):
    settings = _settings(tmp_path)
    pipeline = _native_pipeline_with_no_template_hits(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDEFG")
    first = FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=pipeline,
    ).generate([request])
    assert [artifact.name for artifact in first.written] == ["alpha"]

    second = FeatureBatch(
        settings=settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=pipeline,
    ).generate([request])

    assert second.failures == ()
    assert second.written == ()
    assert [artifact.name for artifact in second.reused] == ["alpha"]


def test_failed_chunk_is_reported_while_later_chunks_complete(tmp_path):
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, max_sequences_per_batch=1)
    batch = FeatureBatch(
        settings=settings,
        mmseqs_process=FirstSequenceFailsMmseqs(),
        af3_pipeline=_native_pipeline_with_no_template_hits(tmp_path),
    )

    result = batch.generate(
        [
            FeatureRequest(name="alpha", sequence="ACDE"),
            FeatureRequest(name="beta", sequence="FGHIK"),
        ]
    )

    assert [(failure.name, failure.error) for failure in result.failures] == [
        ("alpha", "fixture MMseqs2 failure")
    ]
    assert [artifact.name for artifact in result.written] == ["beta"]
    assert not (tmp_path / "features" / "alpha_af3_input.json").exists()
    assert (tmp_path / "features" / "beta_af3_input.json").exists()


def test_all_native_af3_msa_databases_must_be_explicit(tmp_path):
    settings = _settings(tmp_path)
    settings = dataclasses.replace(
        settings, unpaired_databases=settings.unpaired_databases[:2]
    )
    batch = FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=_native_pipeline_with_no_template_hits(tmp_path),
    )

    with pytest.raises(
        ValueError,
        match="uniref90, mgnify, and small_bfd",
    ):
        batch.generate([FeatureRequest(name="alpha", sequence="ACDE")])


def test_database_paths_and_identifiers_must_be_explicit(tmp_path):
    settings = _settings(tmp_path)
    missing_path = dataclasses.replace(
        settings.unpaired_databases[0], path=Path("")
    )
    settings = dataclasses.replace(
        settings,
        unpaired_databases=(missing_path, *settings.unpaired_databases[1:]),
    )
    batch = FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=_native_pipeline_with_no_template_hits(tmp_path),
    )

    with pytest.raises(ValueError, match="path"):
        batch.generate([FeatureRequest(name="alpha", sequence="ACDE")])

    identifier_case = tmp_path / "identifier_case"
    identifier_case.mkdir()
    identifier_settings = _settings(identifier_case)
    blank_identifier = dataclasses.replace(
        identifier_settings.paired_database, identifier=" "
    )
    identifier_settings = dataclasses.replace(
        identifier_settings, paired_database=blank_identifier
    )
    identifier_batch = FeatureBatch(
        settings=identifier_settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=_native_pipeline_with_no_template_hits(identifier_case),
    )

    with pytest.raises(ValueError, match="identifier"):
        identifier_batch.generate([FeatureRequest(name="alpha", sequence="ACDE")])


@pytest.mark.parametrize(
    ("field", "value"),
    (("threads", 0), ("sensitivity", 0), ("e_value", 0)),
)
def test_search_settings_are_validated_before_process_launch(tmp_path, field, value):
    settings = dataclasses.replace(
        _settings(tmp_path), **{field: value}
    )
    batch = FeatureBatch(
        settings=settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=object(),
    )

    with pytest.raises(ValueError, match=field):
        batch.generate([FeatureRequest(name="alpha", sequence="ACDE")])


def test_residue_limit_chunks_unique_sequences_and_keeps_oversized_query_alone(
    tmp_path,
):
    settings = dataclasses.replace(
        _settings(tmp_path),
        max_sequences_per_batch=10,
        max_residues_per_batch=6,
    )
    process = FakeMmseqsProcess()
    batch = FeatureBatch(
        settings=settings,
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    )

    result = batch.generate(
        [
            FeatureRequest(name="alpha", sequence="ACDE"),
            FeatureRequest(name="beta", sequence="FGH"),
            FeatureRequest(name="gamma", sequence="IK"),
            FeatureRequest(name="delta", sequence="LMNPQRS"),
        ]
    )

    assert result.failures == ()
    assert [
        tuple(sequence for _, sequence in records)
        for records in process._queries.values()
    ] == [("ACDE",), ("FGH", "IK"), ("LMNPQRS",)]


def test_changed_database_identifier_regenerates_cached_sequence(tmp_path):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    first = FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])
    assert [artifact.name for artifact in first.written] == ["alpha"]

    changed_database = dataclasses.replace(
        settings.paired_database, identifier="uniprot-2027"
    )
    changed_settings = dataclasses.replace(
        settings, paired_database=changed_database
    )
    process = FakeMmseqsProcess()

    second = FeatureBatch(
        settings=changed_settings,
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert second.reused == ()
    assert [artifact.name for artifact in second.written] == ["alpha"]
    assert len(process._queries) == 1


def test_cached_sequence_supplies_another_name_without_external_search(tmp_path):
    settings = _settings(tmp_path)
    pipeline = PassthroughAf3Pipeline()
    alpha = FeatureRequest(name="alpha", sequence="ACDE")
    FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=pipeline,
    ).generate([alpha])

    result = FeatureBatch(
        settings=settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=pipeline,
    ).generate([alpha, FeatureRequest(name="alias", sequence="ACDE")])

    assert [artifact.name for artifact in result.reused] == ["alpha"]
    assert [artifact.name for artifact in result.written] == ["alias"]
    assert result.failures == ()


def test_compressed_artifact_uses_existing_af3_filename_and_is_cacheable(tmp_path):
    settings = dataclasses.replace(_settings(tmp_path), compress=True)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    artifact = tmp_path / "features" / "alpha_af3_input.json.xz"
    with lzma.open(artifact, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["sequences"][0]["protein"]["sequence"] == "ACDE"

    reused = FeatureBatch(
        settings=settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])
    assert [item.path for item in reused.reused] == [artifact]
