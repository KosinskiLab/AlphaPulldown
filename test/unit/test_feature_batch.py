from __future__ import annotations

import dataclasses
import json
import lzma
from pathlib import Path
import sys

import pytest

try:
    from alphafold3.common import folding_input as _REAL_FOLDING_INPUT
    from alphafold3.cpp import msa_conversion as _REAL_MSA_CONVERSION
    from alphafold3.data import pipeline as _REAL_AF3_PIPELINE
except ImportError as exc:
    pytest.skip(
        f"AlphaFold 3 test dependencies are unavailable: {exc}",
        allow_module_level=True,
    )

from alphapulldown.feature_batch import (
    DatabaseSpec,
    FeatureBatch,
    FeatureBatchSettings,
    FeatureFinalizationSettings,
    FeatureFinalizer,
    FeatureRequest,
    MsaBatch,
    MsaArtifact,
    MsaBatchResult,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
    _normalise_query,
    write_batch_summary,
)
from alphapulldown.utils.feature_metadata import (
    embed_metadata_in_af3_json,
    extract_metadata_from_af3_json,
)


@pytest.fixture(autouse=True)
def _use_real_af3_modules(monkeypatch):
    """Insulate these integration tests from another module's AF3 stubs."""
    common_package = sys.modules["alphafold3.common"]
    cpp_package = sys.modules["alphafold3.cpp"]
    data_package = sys.modules["alphafold3.data"]
    monkeypatch.setitem(
        sys.modules, "alphafold3.common.folding_input", _REAL_FOLDING_INPUT
    )
    monkeypatch.setitem(sys.modules, "alphafold3.data.pipeline", _REAL_AF3_PIPELINE)
    monkeypatch.setitem(
        sys.modules, "alphafold3.cpp.msa_conversion", _REAL_MSA_CONVERSION
    )
    monkeypatch.setattr(
        common_package, "folding_input", _REAL_FOLDING_INPUT, raising=False
    )
    monkeypatch.setattr(data_package, "pipeline", _REAL_AF3_PIPELINE, raising=False)
    monkeypatch.setattr(
        cpp_package, "msa_conversion", _REAL_MSA_CONVERSION, raising=False
    )


class FakeMmseqsProcess:
    """Deterministic stand-in for the external MMseqs2 process."""

    def __init__(self, identity: str = "mmseqs-fixture-1") -> None:
        self._identity = identity
        self.identity_calls = 0
        self._queries: dict[Path, list[tuple[str, str]]] = {}

    def identity(self) -> str:
        self.identity_calls += 1
        return self._identity

    def search_mode(self) -> str:
        return "gpu"

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
        database_index = ("uniref90", "mgnify", "small_bfd", "uniprot").index(
            database_name
        )
        for index, (query_id, sequence) in enumerate(self._queries[query_db]):
            replacement = "VRND"[database_index]
            hit_sequence = replacement + sequence[1:]
            (output_dir / f"{index}.a3m").write_text(
                f">{query_id}\n{sequence}\n>{database_name}_hit\n{hit_sequence}\n",
                encoding="utf-8",
            )


class MissingUnpackOutputMmseqs(FakeMmseqsProcess):
    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        if msa_db.read_text(encoding="utf-8") == "mgnify":
            return
        super().unpack_msa(query_db, msa_db, output_dir)


class EmptyUnpackOutputMmseqs(FakeMmseqsProcess):
    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        if msa_db.read_text(encoding="utf-8") == "mgnify":
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, _ in enumerate(self._queries[query_db]):
                (output_dir / f"{index}.fasta").write_text("", encoding="utf-8")
            return
        super().unpack_msa(query_db, msa_db, output_dir)


class ForbiddenMmseqsProcess:
    def identity(self) -> str:
        return "mmseqs-fixture-1"

    def search_mode(self) -> str:
        return "gpu"

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


class FullHeaderAlignedFastaMmseqs(FakeMmseqsProcess):
    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        del msa_db
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (query_id, sequence) in enumerate(self._queries[query_db]):
            assert sequence == "ACDE"
            (output_dir / f"{index}.fasta").write_text(
                f">{query_id} query description\n"
                "AC-DE\n"
                ">sp|P12345|KINASE_HUMAN Protein kinase OS=Homo sapiens "
                "OX=9606 GN=KIN1\n"
                "ACXDE\n",
                encoding="utf-8",
            )


class PassthroughAf3Pipeline:
    def process(self, fold_input):
        return fold_input


class CountingAf3Pipeline(PassthroughAf3Pipeline):
    def __init__(self):
        self.calls = 0

    def process(self, fold_input):
        self.calls += 1
        return super().process(fold_input)


class FailingAf3Pipeline:
    def process(self, fold_input):
        del fold_input
        raise RuntimeError("fixture template search failure")


def test_subprocess_adapter_identity_comes_from_mmseqs_version(tmp_path):
    binary = tmp_path / "mmseqs"
    binary.write_text(
        "#!/bin/sh\ntest \"$1\" = version\nprintf 'MMseqs2 Version: gpu-build-18\\n'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)

    assert SubprocessMmseqsProcess(binary).identity() == "MMseqs2 Version: gpu-build-18"


def test_subprocess_adapter_requests_aligned_fasta_output(tmp_path):
    binary = tmp_path / "mmseqs"
    arguments = Path(f"{binary}.arguments")
    binary.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "${0}.arguments"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    database = DatabaseSpec(
        name="uniprot", path=tmp_path / "uniprot", identifier="fixture"
    )

    SubprocessMmseqsProcess(binary).result_to_msa(
        tmp_path / "query",
        database,
        tmp_path / "result",
        tmp_path / "msa",
    )

    command = arguments.read_text(encoding="utf-8").splitlines()
    assert command[0] == "result2msa"
    assert command[command.index("--msa-format-mode") + 1] == "2"


def test_gpu_search_does_not_pass_ignored_sensitivity_option(tmp_path):
    binary = tmp_path / "mmseqs"
    arguments = Path(f"{binary}.arguments")
    binary.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "${0}.arguments"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
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
    assert "-s" not in command
    assert command[command.index("--max-seqs") + 1] == "10000"
    assert command[command.index("--gpu") + 1] == "1"


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
        msa_output_dir=tmp_path / "msas",
        temp_dir=tmp_path / "scratch",
        unpaired_databases=tuple(specs[:3]),
        paired_database=specs[3],
        max_sequences_per_batch=8,
        max_residues_per_batch=1_000,
        threads=4,
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
    )


def _finalization_settings(
    settings: FeatureBatchSettings,
) -> FeatureFinalizationSettings:
    return FeatureFinalizationSettings(
        output_dir=settings.output_dir,
        msa_input_dir=settings.msa_output_dir,
        max_template_date=settings.max_template_date,
        template_seqres_database_id=settings.template_seqres_database_id,
        template_mmcif_database_id=settings.template_mmcif_database_id,
        compress=settings.compress,
        base_metadata=settings.base_metadata,
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
        assert ">uniref90_hit" in protein["unpairedMsa"]
        assert ">mgnify_hit" in protein["unpairedMsa"]
        assert ">small_bfd_hit" in protein["unpairedMsa"]
        assert ">uniprot_hit" in protein["pairedMsa"]
        assert protein["templates"] == []


def test_missing_unpack_output_fails_instead_of_caching_query_only_msa(tmp_path):
    settings = _settings(tmp_path)

    result = MsaBatch(
        settings=_msa_settings(settings),
        mmseqs_process=MissingUnpackOutputMmseqs(),
    ).generate([FeatureRequest(name="alpha", sequence="ACDE")])

    assert result.written == ()
    assert [failure.name for failure in result.failures] == ["alpha"]
    assert "mgnify" in result.failures[0].error
    assert not (settings.msa_output_dir / "alpha_mmseqs_msa.json").exists()


def test_empty_unpack_output_fails_instead_of_caching_query_only_msa(tmp_path):
    settings = _settings(tmp_path)

    result = MsaBatch(
        settings=_msa_settings(settings),
        mmseqs_process=EmptyUnpackOutputMmseqs(),
    ).generate([FeatureRequest(name="alpha", sequence="ACDE")])

    assert result.written == ()
    assert [failure.name for failure in result.failures] == ["alpha"]
    assert "mgnify" in result.failures[0].error
    assert "no FASTA records" in result.failures[0].error
    assert not (settings.msa_output_dir / "alpha_mmseqs_msa.json").exists()


def test_normalise_query_rejects_an_empty_alignment():
    with pytest.raises(ValueError, match="contains no records"):
        _normalise_query("", "ACDE")


def test_gpu_msa_and_cpu_finalization_stages_run_independently(tmp_path):
    combined = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    msa_result = MsaBatch(
        settings=_msa_settings(combined),
        mmseqs_process=FakeMmseqsProcess(),
    ).generate([request])
    summary = tmp_path / "summaries" / "shard.json"
    write_batch_summary(summary, msa_result)

    result = FeatureFinalizer(
        settings=_finalization_settings(combined),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert isinstance(msa_result, MsaBatchResult)
    assert isinstance(msa_result.written[0], MsaArtifact)
    assert json.loads(summary.read_text(encoding="utf-8"))["written"] == ["alpha"]
    assert [artifact.name for artifact in result.written] == ["alpha"]
    assert (combined.output_dir / "alpha_af3_input.json").exists()


def test_batch_summary_is_a_content_addressed_artifact_manifest(tmp_path):
    artifact_path = tmp_path / "alpha_mmseqs_msa.json"
    artifact_path.write_bytes(b"bundle\n")
    artifact_stat = artifact_path.stat()
    summary_path = tmp_path / "summaries" / "shard.json"

    write_batch_summary(
        summary_path,
        MsaBatchResult(
            written=(MsaArtifact(name="alpha", path=artifact_path),),
            reused=(),
            failures=(),
        ),
    )

    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "schemaVersion": 2,
        "written": ["alpha"],
        "reused": [],
        "artifacts": [
            {
                "name": "alpha",
                "file": "alpha_mmseqs_msa.json",
                "sizeBytes": 7,
                "mtimeNs": artifact_stat.st_mtime_ns,
                "sha256": (
                    "ef17b7d320f2acc023f2018dab381827ba22f9d01b6c4c97894e1bbfe4928313"
                ),
            }
        ],
    }


def test_deep_stages_accept_only_their_stage_specific_settings(tmp_path):
    combined = _settings(tmp_path)

    with pytest.raises(TypeError, match="MsaBatchSettings"):
        MsaBatch(settings=combined, mmseqs_process=FakeMmseqsProcess())
    with pytest.raises(TypeError, match="FeatureFinalizationSettings"):
        FeatureFinalizer(settings=combined, af3_pipeline=PassthroughAf3Pipeline())


@pytest.mark.parametrize(
    "payload",
    (
        "{",
        "[]",
        json.dumps(
            {
                "sequence": "WRONG",
                "provenance": {},
                "unpairedMsa": ">query\nWRONG\n",
                "pairedMsa": ">query\nWRONG\n",
            }
        ),
        json.dumps(
            {
                "sequence": "ACDE",
                "unpairedMsa": ">query\nACDE\n",
                "pairedMsa": ">query\nACDE\n",
            }
        ),
        json.dumps(
            {
                "sequence": "ACDE",
                "provenance": {},
                "pairedMsa": ">query\nACDE\n",
            }
        ),
    ),
)
def test_invalid_msa_bundle_is_removed_so_the_next_dag_can_repair_it(
    tmp_path, payload
):
    combined = _settings(tmp_path)
    combined.msa_output_dir.mkdir(parents=True)
    bundle = combined.msa_output_dir / "alpha_mmseqs_msa.json"
    bundle.write_text(payload, encoding="utf-8")

    result = FeatureFinalizer(
        settings=_finalization_settings(combined),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([FeatureRequest(name="alpha", sequence="ACDE")])

    assert [failure.name for failure in result.failures] == ["alpha"]
    assert not bundle.exists()


def test_downstream_af3_failure_preserves_a_valid_msa_bundle(tmp_path):
    combined = _settings(tmp_path)
    combined.msa_output_dir.mkdir(parents=True)
    bundle = combined.msa_output_dir / "alpha_mmseqs_msa.json"
    bundle.write_text(
        json.dumps(
            {
                "sequence": "ACDE",
                "provenance": {"schema_version": 4},
                "unpairedMsa": ">query\nACDE\n",
                "pairedMsa": ">query\nACDE\n",
            }
        ),
        encoding="utf-8",
    )

    result = FeatureFinalizer(
        settings=_finalization_settings(combined),
        af3_pipeline=FailingAf3Pipeline(),
    ).generate([FeatureRequest(name="alpha", sequence="ACDE")])

    assert result.failures[0].error == "fixture template search failure"
    assert bundle.exists()


def test_aligned_fasta_conversion_preserves_taxon_header_and_insertions(tmp_path):
    batch = FeatureBatch(
        settings=_settings(tmp_path),
        mmseqs_process=FullHeaderAlignedFastaMmseqs(),
        af3_pipeline=PassthroughAf3Pipeline(),
    )

    result = batch.generate([FeatureRequest(name="alpha", sequence="ACDE")])

    assert result.failures == ()
    payload = json.loads(
        (tmp_path / "features" / "alpha_af3_input.json").read_text(encoding="utf-8")
    )
    paired_msa = payload["sequences"][0]["protein"]["pairedMsa"]
    assert (
        ">sp|P12345|KINASE_HUMAN Protein kinase OS=Homo sapiens "
        "OX=9606 GN=KIN1\n" in paired_msa
    )
    assert "\nACxDE\n" in paired_msa


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
    missing_path = dataclasses.replace(settings.unpaired_databases[0], path=Path(""))
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
    (("threads", 0), ("e_value", 0)),
)
def test_search_settings_are_validated_before_process_launch(tmp_path, field, value):
    settings = dataclasses.replace(_settings(tmp_path), **{field: value})
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
    changed_settings = dataclasses.replace(settings, paired_database=changed_database)
    process = FakeMmseqsProcess()

    second = FeatureBatch(
        settings=changed_settings,
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert second.reused == ()
    assert [artifact.name for artifact in second.written] == ["alpha"]
    assert len(process._queries) == 1


def test_changed_mmseqs_identity_regenerates_cached_sequence(tmp_path):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    first_process = FakeMmseqsProcess(identity="mmseqs-gpu-17")
    first = FeatureBatch(
        settings=settings,
        mmseqs_process=first_process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])
    assert [artifact.name for artifact in first.written] == ["alpha"]

    second_process = FakeMmseqsProcess(identity="mmseqs-gpu-18")
    second = FeatureBatch(
        settings=settings,
        mmseqs_process=second_process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert second.reused == ()
    assert [artifact.name for artifact in second.written] == ["alpha"]
    assert len(second_process._queries) == 1
    assert second_process.identity_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_template_date", "2051-01-01"),
        ("template_seqres_database_id", "pdb-seqres-2051"),
        ("template_mmcif_database_id", "mmcif-2051"),
    ),
)
def test_changed_template_provenance_refinalizes_without_mmseqs_search(
    tmp_path, field, value
):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])
    changed_settings = dataclasses.replace(settings, **{field: value})
    pipeline = CountingAf3Pipeline()

    result = FeatureBatch(
        settings=changed_settings,
        mmseqs_process=ForbiddenMmseqsProcess(),
        af3_pipeline=pipeline,
    ).generate([request])

    assert result.reused == ()
    assert [artifact.name for artifact in result.written] == ["alpha"]
    assert pipeline.calls == 1


def test_broken_mmseqs_identity_invalidates_cache_and_reports_useful_failure(tmp_path):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    MsaBatch(
        settings=_msa_settings(settings), mmseqs_process=FakeMmseqsProcess()
    ).generate([request])

    class BrokenIdentityProcess:
        def identity(self):
            raise RuntimeError("version command failed")

    result = MsaBatch(
        settings=_msa_settings(settings), mmseqs_process=BrokenIdentityProcess()
    ).generate([request])

    assert result.reused == ()
    assert [failure.name for failure in result.failures] == ["alpha"]
    assert (
        "Cannot identify the configured MMseqs2 executable" in result.failures[0].error
    )


def test_stale_final_artifact_reuses_valid_msa_and_only_refinalizes(tmp_path):
    settings = _settings(tmp_path)
    request = FeatureRequest(name="alpha", sequence="ACDE")
    FeatureBatch(
        settings=settings,
        mmseqs_process=FakeMmseqsProcess(),
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])
    artifact = settings.output_dir / "alpha_af3_input.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    metadata = extract_metadata_from_af3_json(payload)[0]
    metadata["other"]["mmseqs2_gpu"]["schema_version"] = 2
    artifact.write_text(
        json.dumps(embed_metadata_in_af3_json(payload, metadata)),
        encoding="utf-8",
    )
    process = FakeMmseqsProcess()

    result = FeatureBatch(
        settings=settings,
        mmseqs_process=process,
        af3_pipeline=PassthroughAf3Pipeline(),
    ).generate([request])

    assert result.reused == ()
    assert [item.name for item in result.written] == ["alpha"]
    assert len(process._queries) == 0


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
