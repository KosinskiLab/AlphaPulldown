"""Foldseek + ESMFold structural templates: parsing, naming, caching, errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphapulldown.structural_templates import (
    FOLDSEEK_OUTPUT_COLUMNS,
    EsmfoldSettings,
    EsmfoldStructurePredictor,
    FoldseekSearchSettings,
    FoldseekTemplateSearcher,
    PredictedStructureCache,
    StructuralTemplateToolMissing,
    StructureDatabaseSpec,
    SubprocessFoldseekProcess,
    alignment_indices,
    parse_foldseek_alignments,
    pdb_chain_name,
    query_sequence_from_a3m,
)


QUERY = "MKTAYIAKQRQISFVKSHFSR"

STRUCTURE = "ATOM      1  N   MET A   1      0.000   0.000   0.000\nEND\n"


def _row(**overrides) -> str:
    fields = {
        "query": "0123456789abcdef",
        "target": "1abc.cif.gz_A",
        "fident": "0.420",
        "alnlen": "11",
        "qstart": "1",
        "qend": "10",
        "tstart": "5",
        "tend": "15",
        "evalue": "1.230E-20",
        "bits": "120",
        "qaln": "MKTAY-IAKQR",
        "taln": "MKTGYQIAKQR",
        "alntmscore": "0.781",
    }
    fields.update(overrides)
    return "\t".join(fields[column] for column in FOLDSEEK_OUTPUT_COLUMNS)


SECOND_ROW = _row(
    target="2xyz_B",
    qstart="3",
    qend="12",
    tstart="1",
    tend="10",
    alnlen="10",
    bits="60",
    qaln="TAYIAKQRQI",
    taln="TAWIAKQRQI",
    alntmscore="0.351",
)


# ------------------------------------------------------------ target naming

@pytest.mark.parametrize(
    "target,expected",
    [
        ("1abc_A", "1abc_A"),
        ("1ABC_A", "1abc_A"),
        ("1abc.cif.gz_A", "1abc_A"),
        ("1abc.pdb_B", "1abc_B"),
        ("pdb1abc.ent.gz_C", "1abc_C"),
        ("1abc-assembly1.cif.gz_A", "1abc_A"),
        ("/databases/pdb/1abc.cif.gz_A", "1abc_A"),
        ("1abc_AAA", "1abc_AAA"),
        ("1abc_.", "1abc_."),
    ],
)
def test_pdb_chain_name_reduces_foldseek_target_names(target, expected):
    assert pdb_chain_name(target) == expected


@pytest.mark.parametrize(
    "target",
    [
        "AF-P12345-F1-model_v4.cif.gz",  # AlphaFold DB model, not a PDB chain
        "1abc",  # no chain
        "notapdbid.cif.gz_A",
        "",
    ],
)
def test_pdb_chain_name_refuses_targets_without_an_mmcif_to_read(target):
    assert pdb_chain_name(target) is None


# -------------------------------------------------------- alignment indices

def test_alignment_indices_are_zero_based_and_mark_gaps():
    indices_query, indices_hit = alignment_indices(
        "MKTAY-IAKQR", "MKTGYQIAKQR", query_start=1, hit_start=5
    )
    assert indices_query == [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9]
    assert indices_hit == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_alignment_indices_reject_rows_of_different_length():
    with pytest.raises(ValueError, match="differ in length"):
        alignment_indices("MKT", "MK", query_start=1, hit_start=1)


def test_alignment_indices_reject_zero_based_starts():
    with pytest.raises(ValueError, match="1-based"):
        alignment_indices("MKT", "MKT", query_start=0, hit_start=1)


# ---------------------------------------------------------- query sequence

def test_query_sequence_is_the_first_a3m_record_without_gaps():
    a3m = ">query\nMKTAY-IAKQR\n>hit\nMKTGYQIAKQR\n"
    assert query_sequence_from_a3m(a3m) == "MKTAYIAKQR"


def test_query_sequence_rejects_an_alignment_without_records():
    with pytest.raises(ValueError, match="no query sequence"):
        query_sequence_from_a3m("")


def test_query_sequence_rejects_a_non_protein_first_record():
    with pytest.raises(ValueError, match="protein sequence"):
        query_sequence_from_a3m(">query\nMKT@YI\n")


# ------------------------------------------------------------- hit parsing

def test_parse_foldseek_alignments_reads_the_requested_columns():
    hits = parse_foldseek_alignments(f"{_row()}\n{SECOND_ROW}\n", QUERY)

    assert [hit.name for hit in hits] == ["1abc_A", "2xyz_B"]
    first = hits[0]
    assert (first.query_start, first.hit_start) == (1, 5)
    assert first.query_alignment == "MKTAY-IAKQR"
    assert first.hit_alignment == "MKTGYQIAKQR"
    # One column is a query gap, so ten columns pair two residues.
    assert first.aligned_cols == 10
    assert first.score == pytest.approx(120.0)
    assert first.alignment_tm_score == pytest.approx(0.781)


def test_parse_foldseek_alignments_skips_a_header_row():
    header = "\t".join(FOLDSEEK_OUTPUT_COLUMNS)
    hits = parse_foldseek_alignments(f"{header}\n{_row()}\n", QUERY)
    assert [hit.name for hit in hits] == ["1abc_A"]


def test_parse_foldseek_alignments_skips_targets_with_no_mmcif_file():
    hits = parse_foldseek_alignments(
        f"{_row(target='AF-P12345-F1-model_v4.cif.gz')}\n{SECOND_ROW}\n", QUERY
    )
    assert [hit.name for hit in hits] == ["2xyz_B"]


def test_parse_foldseek_alignments_skips_rows_with_the_wrong_field_count():
    hits = parse_foldseek_alignments(f"one\ttwo\tthree\n{_row()}\n", QUERY)
    assert [hit.name for hit in hits] == ["1abc_A"]


def test_parse_foldseek_alignments_skips_rows_with_unreadable_numbers():
    hits = parse_foldseek_alignments(f"{_row(qstart='first')}\n{SECOND_ROW}\n", QUERY)
    assert [hit.name for hit in hits] == ["2xyz_B"]


def test_parse_foldseek_alignments_skips_alignments_that_are_not_this_query():
    # The featuriser locates the aligned region by searching the query sequence
    # for it, so a row belonging to another query must not reach it.
    hits = parse_foldseek_alignments(
        f"{_row(qaln='WWWWWWWWWWW')}\n{SECOND_ROW}\n", QUERY
    )
    assert [hit.name for hit in hits] == ["2xyz_B"]


def test_parse_foldseek_alignments_skips_ragged_alignment_rows():
    hits = parse_foldseek_alignments(f"{_row(taln='MKT')}\n{SECOND_ROW}\n", QUERY)
    assert [hit.name for hit in hits] == ["2xyz_B"]


def test_parse_foldseek_alignments_applies_the_tm_score_threshold():
    output = f"{_row()}\n{SECOND_ROW}\n"
    assert [hit.name for hit in parse_foldseek_alignments(output, QUERY)] == [
        "1abc_A",
        "2xyz_B",
    ]
    kept = parse_foldseek_alignments(output, QUERY, min_alignment_tm_score=0.5)
    assert [hit.name for hit in kept] == ["1abc_A"]


def test_parse_foldseek_alignments_tolerates_a_missing_tm_score():
    hits = parse_foldseek_alignments(_row(alntmscore="nan"), QUERY)
    assert hits[0].alignment_tm_score is None


def test_parse_foldseek_alignments_ignores_blank_and_comment_lines():
    hits = parse_foldseek_alignments(f"\n# a comment\n{_row()}\n\n", QUERY)
    assert len(hits) == 1


# ------------------------------------------------- AlphaFold 2 hit handover

def test_hit_converts_to_an_alphafold_template_hit():
    parsers = pytest.importorskip("alphafold.data.parsers")
    hit = parse_foldseek_alignments(_row(), QUERY)[0].to_template_hit()

    assert isinstance(hit, parsers.TemplateHit)
    # AlphaFold 2 parses the PDB id and chain straight off the front of the name.
    assert hit.name == "1abc_A"
    assert hit.aligned_cols == 10
    assert hit.indices_query == [0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9]
    assert hit.indices_hit == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_converted_hit_maps_query_residues_onto_the_template():
    templates = pytest.importorskip("alphafold.data.templates")
    hit = parse_foldseek_alignments(_row(), QUERY)[0].to_template_hit()

    mapping = templates._build_query_to_hit_index_mapping(
        hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query, QUERY
    )

    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}


def test_alphafold_can_read_the_pdb_id_and_chain_from_every_hit_name():
    templates = pytest.importorskip("alphafold.data.templates")
    hits = parse_foldseek_alignments(f"{_row()}\n{SECOND_ROW}\n", QUERY)

    parsed = [
        templates._get_pdb_id_and_chain(hit.to_template_hit()) for hit in hits
    ]

    assert parsed == [("1abc", "A"), ("2xyz", "B")]


# -------------------------------------------------------------- test doubles

class _FakePredictor:
    def __init__(self, identity: str = "esmfold:test:0123456789abcdef"):
        self._identity = identity
        self.folded: list[str] = []

    def identity(self) -> str:
        return self._identity

    def predict(self, sequence: str) -> str:
        self.folded.append(sequence)
        return STRUCTURE


class _FakeFoldseek:
    def __init__(self, output: str, identity: str = "foldseek 9.427df8a"):
        self._output = output
        self._identity = identity
        self.searches: list[tuple[str, str]] = []

    def identity(self) -> str:
        return self._identity

    def search(self, query_structure: Path, settings: FoldseekSearchSettings) -> str:
        self.searches.append(
            (Path(query_structure).name, Path(query_structure).read_text())
        )
        assert settings.database.identifier
        return self._output


def _settings(tmp_path: Path, *, identifier: str = "pdb-2024-01", **overrides):
    defaults = dict(
        database=StructureDatabaseSpec(
            name="foldseek", path=tmp_path / "db", identifier=identifier
        ),
        temp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
    )
    defaults.update(overrides)
    return FoldseekSearchSettings(**defaults)


def _searcher(tmp_path: Path, predictor, foldseek, **overrides):
    return FoldseekTemplateSearcher(
        settings=_settings(tmp_path, **overrides),
        structures=PredictedStructureCache(
            cache_dir=_settings(tmp_path, **overrides).cache_dir, predictor=predictor
        ),
        foldseek_process=foldseek,
    )


# ---------------------------------------------------------------- searching

def test_searcher_announces_the_formats_alphafold_asks_about(tmp_path):
    searcher = _searcher(tmp_path, _FakePredictor(), _FakeFoldseek(_row()))
    assert searcher.input_format == "a3m"
    assert searcher.output_format == "m8"


def test_query_folds_the_first_a3m_record_and_returns_foldseek_output(tmp_path):
    predictor = _FakePredictor()
    foldseek = _FakeFoldseek(_row())
    searcher = _searcher(tmp_path, predictor, foldseek)

    output = searcher.query(f">query\n{QUERY}\n>hit\n{QUERY}\n")

    assert output == _row()
    assert predictor.folded == [QUERY]
    name, structure = foldseek.searches[0]
    assert name.endswith(".pdb")
    assert structure == STRUCTURE


def test_a_repeated_query_reuses_both_caches(tmp_path):
    predictor = _FakePredictor()
    foldseek = _FakeFoldseek(_row())
    a3m = f">query\n{QUERY}\n"

    first = _searcher(tmp_path, predictor, foldseek).query(a3m)
    second = _searcher(tmp_path, predictor, foldseek).query(a3m)

    assert first == second
    assert predictor.folded == [QUERY]
    assert len(foldseek.searches) == 1


def test_a_rebuilt_database_invalidates_alignments_but_not_the_structure(tmp_path):
    predictor = _FakePredictor()
    foldseek = _FakeFoldseek(_row())
    a3m = f">query\n{QUERY}\n"

    _searcher(tmp_path, predictor, foldseek, identifier="pdb-2024-01").query(a3m)
    _searcher(tmp_path, predictor, foldseek, identifier="pdb-2025-06").query(a3m)

    assert predictor.folded == [QUERY]
    assert len(foldseek.searches) == 2


def test_new_weights_invalidate_the_cached_structure(tmp_path):
    foldseek = _FakeFoldseek(_row())
    a3m = f">query\n{QUERY}\n"

    _searcher(tmp_path, _FakePredictor("esmfold:v1"), foldseek).query(a3m)
    later = _FakePredictor("esmfold:v2")
    _searcher(tmp_path, later, foldseek).query(a3m)

    assert later.folded == [QUERY]
    assert len(foldseek.searches) == 2


def test_a_corrupt_cache_entry_is_replaced_rather_than_trusted(tmp_path):
    predictor = _FakePredictor()
    foldseek = _FakeFoldseek(_row())
    a3m = f">query\n{QUERY}\n"
    searcher = _searcher(tmp_path, predictor, foldseek)
    searcher.query(a3m)
    for entry in (tmp_path / "cache").glob("*.json"):
        entry.write_text("{ this is not json")

    assert _searcher(tmp_path, predictor, foldseek).query(a3m) == _row()
    assert predictor.folded == [QUERY, QUERY]


def test_alignment_provenance_records_the_database_and_the_settings(tmp_path):
    searcher = _searcher(tmp_path, _FakePredictor(), _FakeFoldseek(_row()))

    provenance = searcher.provenance()

    assert provenance["database"]["identifier"] == "pdb-2024-01"
    assert provenance["foldseek"] == "foldseek 9.427df8a"
    assert provenance["predictor"] == "esmfold:test:0123456789abcdef"
    assert provenance["columns"] == list(FOLDSEEK_OUTPUT_COLUMNS)


def test_a_published_cache_entry_carries_its_provenance(tmp_path):
    searcher = _searcher(tmp_path, _FakePredictor(), _FakeFoldseek(_row()))
    searcher.query(f">query\n{QUERY}\n")

    alignments = json.loads(
        next((tmp_path / "cache").glob("*_foldseek.json")).read_text()
    )

    assert alignments["sequence"] == QUERY
    assert alignments["provenance"] == searcher.provenance()


def test_get_template_hits_returns_alphafold_hits(tmp_path):
    pytest.importorskip("alphafold.data.parsers")
    searcher = _searcher(tmp_path, _FakePredictor(), _FakeFoldseek(_row()))

    hits = searcher.get_template_hits(_row(), QUERY)

    assert [hit.name for hit in hits] == ["1abc_A"]


# ------------------------------------------------------- structure caching

def test_structure_cache_reports_whether_a_sequence_is_already_folded(tmp_path):
    predictor = _FakePredictor()
    cache = PredictedStructureCache(cache_dir=tmp_path / "cache", predictor=predictor)

    assert cache.cached(QUERY) is False
    cache.structure(QUERY)
    assert cache.cached(QUERY) is True
    assert cache.structure(QUERY) == STRUCTURE
    assert predictor.folded == [QUERY]


def test_structure_cache_refuses_a_non_protein_sequence(tmp_path):
    cache = PredictedStructureCache(
        cache_dir=tmp_path / "cache", predictor=_FakePredictor()
    )
    with pytest.raises(ValueError, match="protein sequence"):
        cache.structure("MKT@YI")


# ------------------------------------------------------- settings validation

@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"e_value": 0.0}, "e_value"),
        ({"max_hits": 0}, "max_hits"),
        ({"threads": 0}, "threads"),
        ({"min_alignment_tm_score": 1.5}, "min_alignment_tm_score"),
        ({"alignment_type": 7}, "alignment_type"),
    ],
)
def test_unusable_search_settings_are_refused(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        FoldseekTemplateSearcher(
            settings=_settings(tmp_path, **overrides),
            structures=PredictedStructureCache(
                cache_dir=tmp_path, predictor=_FakePredictor()
            ),
            foldseek_process=_FakeFoldseek(_row()),
        )


def test_a_database_without_an_identifier_is_refused(tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        FoldseekTemplateSearcher(
            settings=_settings(tmp_path, identifier="  "),
            structures=PredictedStructureCache(
                cache_dir=tmp_path, predictor=_FakePredictor()
            ),
            foldseek_process=_FakeFoldseek(_row()),
        )


# ------------------------------------------------------------ missing tools

def test_an_unconfigured_foldseek_says_so_instead_of_failing_obscurely():
    with pytest.raises(StructuralTemplateToolMissing, match="Install Foldseek"):
        SubprocessFoldseekProcess(None).identity()


def test_a_foldseek_path_that_does_not_exist_says_so(tmp_path):
    process = SubprocessFoldseekProcess(tmp_path / "no-such-foldseek")
    with pytest.raises(StructuralTemplateToolMissing, match="not found"):
        process.identity()


def test_missing_esmfold_weights_name_the_directory(tmp_path):
    predictor = EsmfoldStructurePredictor(
        EsmfoldSettings(model_dir=tmp_path / "weights", device="cpu")
    )
    with pytest.raises(StructuralTemplateToolMissing, match="No ESMFold weight files"):
        predictor.identity()


def test_esmfold_identity_witnesses_the_checkpoint_contents(tmp_path):
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"0" * 32)
    predictor = EsmfoldStructurePredictor(
        EsmfoldSettings(model_dir=model_dir, device="cpu")
    )

    before = predictor.identity()
    (model_dir / "model.safetensors").write_bytes(b"0" * 64)

    assert before.startswith("esmfold:weights:")
    assert predictor.identity() != before


def test_esmfold_refuses_a_sequence_longer_than_the_configured_limit(tmp_path):
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"0")
    predictor = EsmfoldStructurePredictor(
        EsmfoldSettings(model_dir=model_dir, device="cpu", max_sequence_length=5)
    )
    with pytest.raises(ValueError, match="exceeds the configured"):
        predictor.predict("MKTAYIAKQR")
