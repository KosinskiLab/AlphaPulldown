"""Structural template search against a real Foldseek binary and a real database.

The unit and integration tiers stub Foldseek: they pin the command line and the
parser against canned rows. Neither can tell whether Foldseek actually accepts
the columns AlphaPulldown asks for, or whether the numbers it really writes mean
what the parser assumes they mean. That is what this tier is for.

A wrong alignment offset is the failure worth guarding against here, because it
does not raise -- it produces silently wrong templates. So the query is a
*truncated* copy of a chain in the database: the self-hit must come back with the
query starting at residue 1 and the template starting at the truncation point,
and every aligned column is checked against both full sequences.

ESMFold is not involved. Folding is stubbed with a structure taken from the test
data, so this stays deterministic and CPU-only; the GPU chain is smoke-tested in
``test/cluster/check_structural_templates.py``.

Set ``FOLDSEEK_BINARY`` (or put ``foldseek`` on PATH) to run it.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import warnings

import pytest

pytestmark = [pytest.mark.functional, pytest.mark.external_tools]

from alphapulldown.structural_templates import (  # noqa: E402
    FOLDSEEK_OUTPUT_COLUMNS,
    FoldseekSearchSettings,
    FoldseekTemplateSearcher,
    PredictedStructureCache,
    StructureDatabaseSpec,
    SubprocessFoldseekProcess,
    parse_foldseek_alignments,
    pdb_chain_name,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
SOURCE_MMCIF = TEST_DATA / "templates" / "3L4Q.cif"
OTHER_MMCIF = TEST_DATA / "templates" / "0099.cif"

# The query is chain A with this many N-terminal residues removed, so the
# self-hit's template start is not 1 and an off-by-one cannot hide.
TRUNCATION = 10
QUERY_CHAIN = "A"
# A copy of the same structure under a name that is not a PDB chain. It hits, so
# the "skipped, no mmCIF to read" path is genuinely exercised rather than assumed.
NON_PDB_STEM = "AF-P12345-F1-model_v4"


def _foldseek_binary() -> str:
    binary = os.environ.get("FOLDSEEK_BINARY") or shutil.which("foldseek")
    if not binary:
        pytest.skip(
            "Foldseek is not installed; set FOLDSEEK_BINARY or put foldseek on PATH"
        )
    if not Path(binary).is_file():
        pytest.skip(f"FOLDSEEK_BINARY does not exist: {binary}")
    return binary


def _chain_residues(path: Path, chain_id: str):
    from Bio.PDB import MMCIFParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = MMCIFParser(QUIET=True).get_structure("query", str(path))
    model = next(iter(structure))
    return structure, model, [
        residue for residue in model[chain_id] if residue.id[0] == " "
    ]


def _one_letter(residue) -> str:
    from Bio.PDB.Polypeptide import index_to_one, three_to_index

    try:
        return index_to_one(three_to_index(residue.get_resname()))
    except (KeyError, IndexError):
        return "X"


@pytest.fixture(scope="module")
def foldseek_binary() -> str:
    return _foldseek_binary()


@pytest.fixture(scope="module")
def chain_sequence() -> str:
    """The full one-letter sequence of the chain the database entry contains."""
    _, _, residues = _chain_residues(SOURCE_MMCIF, QUERY_CHAIN)
    return "".join(_one_letter(residue) for residue in residues)


@pytest.fixture(scope="module")
def query(tmp_path_factory) -> tuple[str, str]:
    """A truncated copy of the chain: its PDB text and its exact sequence."""
    from Bio.PDB import PDBIO, Select

    structure, model, residues = _chain_residues(SOURCE_MMCIF, QUERY_CHAIN)
    keep = {residue.id for residue in residues[TRUNCATION:]}

    class _Truncated(Select):
        def accept_model(self, candidate):
            return candidate.id == model.id

        def accept_chain(self, candidate):
            return candidate.id == QUERY_CHAIN

        def accept_residue(self, candidate):
            return candidate.id in keep

    writer = PDBIO()
    writer.set_structure(structure)
    path = tmp_path_factory.mktemp("query") / "truncated.pdb"
    writer.save(str(path), select=_Truncated())
    sequence = "".join(_one_letter(residue) for residue in residues[TRUNCATION:])
    return path.read_text(encoding="utf-8"), sequence


@pytest.fixture(scope="module")
def mmcif_dir(tmp_path_factory) -> Path:
    """An mmCIF directory named the way AlphaFold 2 expects to read it."""
    directory = tmp_path_factory.mktemp("mmcif_files")
    shutil.copy(SOURCE_MMCIF, directory / "3l4q.cif")
    shutil.copy(OTHER_MMCIF, directory / "0099.cif")
    return directory


@pytest.fixture(scope="module")
def structure_database(foldseek_binary, tmp_path_factory, mmcif_dir) -> Path:
    """A tiny Foldseek database built from those same mmCIF files."""
    source = tmp_path_factory.mktemp("db_source")
    for entry in mmcif_dir.glob("*.cif"):
        shutil.copy(entry, source / entry.name)
    shutil.copy(SOURCE_MMCIF, source / f"{NON_PDB_STEM}.cif")

    database = tmp_path_factory.mktemp("foldseek_db") / "structures"
    completed = subprocess.run(
        [foldseek_binary, "createdb", str(source), str(database)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode:
        pytest.fail(f"foldseek createdb failed:\n{completed.stderr[-2000:]}")
    return database


@pytest.fixture(scope="module")
def search_output(foldseek_binary, structure_database, query, tmp_path_factory) -> str:
    """Raw Foldseek output, produced through AlphaPulldown's own searcher."""
    structure, sequence = query
    root = tmp_path_factory.mktemp("search")

    class _RecordedStructure:
        """Stands in for ESMFold: this tier is about Foldseek, not folding."""

        def identity(self) -> str:
            return "test-data-structure:3L4Q_A"

        def predict(self, _sequence: str) -> str:
            return structure

    searcher = FoldseekTemplateSearcher(
        settings=FoldseekSearchSettings(
            database=StructureDatabaseSpec(
                name="foldseek",
                path=structure_database,
                identifier="functional-test-db",
            ),
            temp_dir=root / "scratch",
            cache_dir=root / "cache",
            e_value=1.0,
            max_hits=50,
            threads=2,
        ),
        structures=PredictedStructureCache(
            cache_dir=root / "cache", predictor=_RecordedStructure()
        ),
        foldseek_process=SubprocessFoldseekProcess(foldseek_binary),
    )
    return searcher.query(f">query\n{sequence}\n")


@pytest.fixture(scope="module")
def hits(search_output, query):
    _, sequence = query
    return parse_foldseek_alignments(search_output, sequence)


def test_foldseek_accepts_every_column_alphapulldown_parses(search_output):
    """The requested --format-output columns all exist in this Foldseek build."""
    rows = [line for line in search_output.splitlines() if line.strip()]
    assert rows, "the real search returned no rows at all"
    for row in rows:
        assert len(row.split("\t")) == len(FOLDSEEK_OUTPUT_COLUMNS), row


def test_real_output_parses_into_usable_hits(hits):
    assert hits, "no hit survived parsing of real Foldseek output"
    # The database was built from the query's own structure, so it must be found.
    assert "3l4q_A" in {hit.name for hit in hits}
    for hit in hits:
        assert re.fullmatch(r"[0-9a-z]{4}_[A-Za-z0-9.]+", hit.name), hit.name
        assert hit.query_alignment and hit.hit_alignment
        assert len(hit.query_alignment) == len(hit.hit_alignment)
        assert hit.aligned_cols > 0


def test_every_hit_names_an_mmcif_file_the_featuriser_can_open(hits, mmcif_dir):
    templates = pytest.importorskip("alphafold.data.templates")

    for hit in hits:
        pdb_id, chain_id = templates._get_pdb_id_and_chain(hit.to_template_hit())
        assert chain_id
        assert (mmcif_dir / f"{pdb_id}.cif").is_file(), (
            f"hit {hit.name} has no mmCIF file in the template directory"
        )


def test_a_target_that_is_not_a_pdb_chain_is_dropped(search_output, hits):
    targets = [line.split("\t")[1] for line in search_output.splitlines() if line.strip()]
    non_pdb = [target for target in targets if target.startswith(NON_PDB_STEM)]
    assert non_pdb, (
        "the non-PDB entry did not appear in the raw output, so this test would "
        "not have exercised the skip path"
    )
    assert all(pdb_chain_name(target) is None for target in non_pdb)
    assert not [hit for hit in hits if NON_PDB_STEM.lower() in hit.name.lower()]


def test_alignment_offsets_place_every_residue_where_it_belongs(
    hits, query, chain_sequence
):
    """The check a wrong offset would fail silently everywhere else.

    The query is the database chain minus its first ``TRUNCATION`` residues, so
    the self-hit has to start at query residue 1 and template residue
    ``TRUNCATION + 1``. Both alignment rows are then checked residue by residue
    against the sequences they claim to index.
    """
    _, sequence = query
    self_hit = next(hit for hit in hits if hit.name == "3l4q_A")

    assert self_hit.query_start == 1
    assert self_hit.hit_start == TRUNCATION + 1

    template_hit = self_hit.to_template_hit()
    for column, (query_index, hit_index) in enumerate(
        zip(template_hit.indices_query, template_hit.indices_hit)
    ):
        if query_index != -1:
            assert sequence[query_index] == template_hit.query[column]
        if hit_index != -1:
            assert chain_sequence[hit_index] == template_hit.hit_sequence[column]

    assert template_hit.indices_query[0] == 0
    assert template_hit.indices_hit[0] == TRUNCATION


def test_alphafold_maps_the_whole_query_onto_the_self_hit(hits, query):
    templates = pytest.importorskip("alphafold.data.templates")
    _, sequence = query
    template_hit = next(
        hit for hit in hits if hit.name == "3l4q_A"
    ).to_template_hit()

    mapping = templates._build_query_to_hit_index_mapping(
        template_hit.query,
        template_hit.hit_sequence,
        template_hit.indices_hit,
        template_hit.indices_query,
        sequence,
    )

    # A gapless self-alignment: every query residue maps, in order, onto the
    # aligned template residue at the same position.
    assert mapping == {index: index for index in range(len(sequence))}


def test_the_search_result_is_cached_for_the_next_run(
    foldseek_binary, structure_database, query, tmp_path
):
    """A second identical query must not shell out to Foldseek again."""
    structure, sequence = query
    calls: list[str] = []

    class _CountingFoldseek(SubprocessFoldseekProcess):
        def search(self, query_structure, settings):
            calls.append(str(query_structure))
            return super().search(query_structure, settings)

    def _searcher():
        return FoldseekTemplateSearcher(
            settings=FoldseekSearchSettings(
                database=StructureDatabaseSpec(
                    name="foldseek",
                    path=structure_database,
                    identifier="functional-test-db",
                ),
                temp_dir=tmp_path / "scratch",
                cache_dir=tmp_path / "cache",
                e_value=1.0,
                max_hits=50,
                threads=2,
            ),
            structures=PredictedStructureCache(
                cache_dir=tmp_path / "cache",
                predictor=type(
                    "_Stub",
                    (),
                    {
                        "identity": lambda self: "test-data-structure:3L4Q_A",
                        "predict": lambda self, _sequence: structure,
                    },
                )(),
            ),
            foldseek_process=_CountingFoldseek(foldseek_binary),
        )

    first = _searcher().query(f">query\n{sequence}\n")
    second = _searcher().query(f">query\n{sequence}\n")

    assert first == second
    assert len(calls) == 1
