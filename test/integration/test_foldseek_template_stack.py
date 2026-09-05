"""The Foldseek template path as the feature CLI wires it, against a stub binary.

Nothing here needs a GPU, real weights, a real Foldseek build or a structure
database: the executable is a script the test writes, and the structure predictor
is a double. What is exercised is the wiring -- the command AlphaPulldown builds,
the searcher AlphaFold 2 is handed, and the promise that a default run is
untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.integration

from alphapulldown.scripts._foldseek_cli import (  # noqa: E402
    STRUCTURAL_TEMPLATE_FLAG_NAMES,
)
from alphapulldown.structural_templates import (  # noqa: E402
    FOLDSEEK_OUTPUT_COLUMNS,
    FoldseekSearchSettings,
    FoldseekTemplateSearcher,
    PredictedStructureCache,
    StructureDatabaseSpec,
    SubprocessFoldseekProcess,
)

QUERY = "MKTAYIAKQRQISFVKSHFSR"

ALIGNMENT_ROW = "\t".join(
    {
        "query": "0123456789abcdef",
        "target": "1abc.cif.gz_A",
        "fident": "0.42",
        "alnlen": "10",
        "qstart": "1",
        "qend": "10",
        "tstart": "5",
        "tend": "14",
        "evalue": "1e-20",
        "bits": "120",
        "qaln": "MKTAYIAKQR",
        "taln": "MKTGYIAKQR",
        "alntmscore": "0.78",
    }[column]
    for column in FOLDSEEK_OUTPUT_COLUMNS
)


class _StubPredictor:
    """Stands in for ESMFold; the point here is the Foldseek command."""

    def identity(self) -> str:
        return "stub-predictor:1"

    def predict(self, sequence: str) -> str:
        return f"REMARK stub structure for {len(sequence)} residues\nEND\n"


def _stub_foldseek(tmp_path: Path, record: Path) -> Path:
    """A fake ``foldseek`` that records its arguments and writes canned output."""
    binary = tmp_path / "foldseek"
    binary.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import pathlib, sys",
                "argv = sys.argv[1:]",
                "if argv[:1] == ['version']:",
                "    print('9.427df8a')",
                "    raise SystemExit(0)",
                f"pathlib.Path({str(record)!r}).write_text('\\n'.join(argv))",
                # easy-search <query> <database> <output> <tmp>
                f"pathlib.Path(argv[3]).write_text({ALIGNMENT_ROW + chr(10)!r})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _searcher(tmp_path: Path, binary: Path) -> FoldseekTemplateSearcher:
    return FoldseekTemplateSearcher(
        settings=FoldseekSearchSettings(
            database=StructureDatabaseSpec(
                name="foldseek",
                path=tmp_path / "db" / "pdb",
                identifier="pdb-2024-01",
            ),
            temp_dir=tmp_path / "scratch",
            cache_dir=tmp_path / "cache",
            e_value=1e-5,
            max_hits=42,
            threads=3,
        ),
        structures=PredictedStructureCache(
            cache_dir=tmp_path / "cache", predictor=_StubPredictor()
        ),
        foldseek_process=SubprocessFoldseekProcess(binary),
    )


def test_the_search_command_asks_foldseek_for_the_columns_it_parses(tmp_path):
    record = tmp_path / "argv.txt"
    searcher = _searcher(tmp_path, _stub_foldseek(tmp_path, record))

    output = searcher.query(f">query\n{QUERY}\n")

    assert output.strip() == ALIGNMENT_ROW
    argv = record.read_text(encoding="utf-8").splitlines()
    # easy-search <query structure> <database> <output> <scratch> then flag pairs.
    assert argv[0] == "easy-search"
    assert argv[1].endswith(".pdb")
    assert argv[2] == str(tmp_path / "db" / "pdb")
    options = dict(zip(argv[5::2], argv[6::2]))
    assert options["--format-output"] == ",".join(FOLDSEEK_OUTPUT_COLUMNS)
    assert options["-e"] == "1e-05"
    assert options["--max-seqs"] == "42"
    assert options["--threads"] == "3"
    assert options["--alignment-type"] == "2"


def test_the_hits_reach_alphafold_in_the_form_its_featuriser_expects(tmp_path):
    templates = pytest.importorskip("alphafold.data.templates")
    record = tmp_path / "argv.txt"
    searcher = _searcher(tmp_path, _stub_foldseek(tmp_path, record))

    hits = searcher.get_template_hits(searcher.query(f">query\n{QUERY}\n"), QUERY)

    assert [templates._get_pdb_id_and_chain(hit) for hit in hits] == [("1abc", "A")]


def test_a_failing_foldseek_reports_its_own_error(tmp_path):
    binary = tmp_path / "foldseek"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "sys.stderr.write('Invalid database\\n')\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    searcher = _searcher(tmp_path, binary)

    with pytest.raises(RuntimeError, match="Invalid database"):
        searcher.query(f">query\n{QUERY}\n")


# ------------------------------------------ the default path must not change

@pytest.fixture()
def feature_cli():
    return pytest.importorskip("alphapulldown.scripts.create_individual_features")


def _configure_template_flags(flag_values, monkeypatch, tmp_path: Path) -> None:
    mmcif_dir = tmp_path / "mmcif"
    mmcif_dir.mkdir(exist_ok=True)
    (mmcif_dir / "1abc.cif").write_text("data_1abc\n", encoding="utf-8")
    seqres = tmp_path / "pdb_seqres.txt"
    seqres.write_text(">1abc_A\nMKT\n", encoding="utf-8")
    for name, value in {
        "template_mmcif_dir": str(mmcif_dir),
        "pdb_seqres_database_path": str(seqres),
        "max_template_date": "2024-01-01",
        "obsolete_pdbs_path": None,
        "output_dir": str(tmp_path / "features"),
    }.items():
        monkeypatch.setattr(flag_values, name, value)


def test_the_feature_cli_leaves_the_sequence_search_alone_by_default(
    feature_cli, tmp_flags, tmp_path, monkeypatch
):
    hmmsearch = pytest.importorskip("alphafold.data.tools.hmmsearch")
    _configure_template_flags(tmp_flags, monkeypatch, tmp_path)

    assert feature_cli.FLAGS["use_foldseek_templates"].default is False
    assert tmp_flags.use_foldseek_templates is False
    searcher, featuriser = feature_cli._create_af2_template_stack()

    assert isinstance(searcher, hmmsearch.Hmmsearch)
    assert searcher.input_format == "sto"
    assert type(featuriser).__name__ == "HmmsearchHitFeaturizer"


def test_the_feature_cli_swaps_in_the_structural_searcher_when_asked(
    feature_cli, tmp_flags, tmp_path, monkeypatch
):
    pytest.importorskip("alphafold.data.templates")
    _configure_template_flags(tmp_flags, monkeypatch, tmp_path)
    monkeypatch.setattr(tmp_flags, "use_foldseek_templates", True)
    monkeypatch.setattr(tmp_flags, "foldseek_database_path", str(tmp_path / "db"))
    monkeypatch.setattr(tmp_flags, "foldseek_database_id", "pdb-2024-01")
    monkeypatch.setattr(tmp_flags, "esmfold_model_dir", str(tmp_path / "weights"))

    searcher, featuriser = feature_cli._create_af2_template_stack()

    assert isinstance(searcher, FoldseekTemplateSearcher)
    # Featurisation is unchanged; only the source of the hits differs.
    assert type(featuriser).__name__ == "HhsearchHitFeaturizer"


def test_enabling_it_without_a_database_fails_before_any_work(
    feature_cli, tmp_flags, tmp_path, monkeypatch
):
    _configure_template_flags(tmp_flags, monkeypatch, tmp_path)
    monkeypatch.setattr(tmp_flags, "use_foldseek_templates", True)
    monkeypatch.setattr(tmp_flags, "foldseek_database_path", None)

    with pytest.raises(ValueError, match="--foldseek_database_path"):
        feature_cli.validate_data_pipeline_flags()


def test_a_default_run_records_no_structural_template_metadata(
    feature_cli, tmp_flags, monkeypatch
):
    # A Foldseek binary on PATH must not turn up in the provenance of features
    # that were never searched with it.
    monkeypatch.setattr(tmp_flags, "foldseek_binary_path", "/usr/local/bin/foldseek")

    flag_dict = tmp_flags.flag_values_dict()
    filtered = feature_cli.structural_template_metadata_flags(flag_dict)

    assert filtered == {
        name: value
        for name, value in flag_dict.items()
        if name not in STRUCTURAL_TEMPLATE_FLAG_NAMES
    }
    assert "foldseek_binary_path" not in filtered
