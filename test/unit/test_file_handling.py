from pathlib import Path

import pytest

from alphapulldown.utils import file_handling


def test_temp_fasta_file_writes_contents_and_cleans_up():
    with file_handling.temp_fasta_file(">seq\nACDE\n") as fasta_path:
        path = Path(fasta_path)
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == ">seq\nACDE\n"

    assert not path.exists()


def test_convert_fasta_description_to_protein_name_sanitizes_symbols():
    protein_name = file_handling.convert_fasta_description_to_protein_name(
        ">sp|P12345|Protein A:chain;1?"
    )

    assert protein_name == "sp_P12345_Protein_A_chain_1_"


def test_parse_fasta_parses_multiple_sequences_and_skips_blank_lines():
    sequences, descriptions = file_handling.parse_fasta(
        ">protein one\nACD\n\nEF\n>protein|two\nGHI\n"
    )

    assert sequences == ["ACDEF", "GHI"]
    assert descriptions == ["protein_one", "protein_two"]


def test_iter_seqs_yields_sequences_from_multiple_files(tmp_path):
    fasta_a = tmp_path / "a.fasta"
    fasta_b = tmp_path / "b.fasta"
    fasta_a.write_text(">first\nAAA\n", encoding="utf-8")
    fasta_b.write_text(">second\nBBB\n", encoding="utf-8")

    records = list(file_handling.iter_seqs([str(fasta_a), str(fasta_b)]))

    assert records == [("AAA", "first"), ("BBB", "second")]


def test_make_dir_monomer_dictionary_maps_files_to_their_source_dirs(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "proteinA.pkl").write_text("", encoding="utf-8")
    (dir_b / "proteinB.pkl.xz").write_text("", encoding="utf-8")

    result = file_handling.make_dir_monomer_dictionary([str(dir_a), str(dir_b)])

    assert result == {
        "proteinA.pkl": str(dir_a),
        "proteinB.pkl.xz": str(dir_b),
    }


def test_parse_csv_file_raises_for_missing_fasta():
    with pytest.raises(FileNotFoundError):
        file_handling.parse_csv_file("missing.csv", ["missing.fasta"], "/templates")


def test_parse_csv_file_returns_unique_records_without_clustering(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">proteinA\nACDE\n", encoding="utf-8")
    csv_path = tmp_path / "description.csv"
    csv_path.write_text(
        "proteinA,template1.cif,A\nproteinA,template2.cif,B\ninvalid,row\nmissing,template3.cif,C\n",
        encoding="utf-8",
    )

    result = file_handling.parse_csv_file(
        str(csv_path), [str(fasta)], str(tmp_path / "templates"), cluster=False
    )

    assert result == [
        {
            "protein": "proteinA.template1.cif.A",
            "sequence": "ACDE",
            "templates": [str(tmp_path / "templates" / "template1.cif")],
            "chains": ["A"],
        },
        {
            "protein": "proteinA.template2.cif.B",
            "sequence": "ACDE",
            "templates": [str(tmp_path / "templates" / "template2.cif")],
            "chains": ["B"],
        },
    ]


def test_parse_csv_file_clusters_multiple_templates_per_protein(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">proteinA\nACDE\n", encoding="utf-8")
    csv_path = tmp_path / "description.csv"
    csv_path.write_text(
        "proteinA,template1.cif,A\nproteinA,template2.cif,B\n",
        encoding="utf-8",
    )

    result = file_handling.parse_csv_file(
        str(csv_path), [str(fasta)], str(tmp_path / "templates"), cluster=True
    )

    assert result == [
        {
            "protein": "proteinA",
            "sequence": "ACDE",
            "templates": [
                str(tmp_path / "templates" / "template1.cif"),
                str(tmp_path / "templates" / "template2.cif"),
            ],
            "chains": ["A", "B"],
        }
    ]
