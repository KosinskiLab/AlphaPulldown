import os
import pickle
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


# open_atomic: outputs a resumed run relies on appear whole or not at all.

def _leftovers(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))


def test_open_atomic_replaces_the_file_with_the_whole_new_content(tmp_path):
    target = tmp_path / "timings.json"
    target.write_text("old", encoding="utf-8")

    with file_handling.open_atomic(target) as handle:
        handle.write("new")

    assert target.read_text(encoding="utf-8") == "new"
    assert _leftovers(tmp_path) == []


def test_open_atomic_keeps_the_previous_file_when_writing_fails(tmp_path):
    target = tmp_path / "unrelaxed_model.pdb"
    target.write_text("complete model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="killed"):
        with file_handling.open_atomic(target) as handle:
            handle.write("half a mod")
            raise RuntimeError("killed")

    assert target.read_text(encoding="utf-8") == "complete model"
    assert _leftovers(tmp_path) == []


def test_open_atomic_leaves_nothing_when_a_new_file_fails(tmp_path):
    target = tmp_path / "result_model.pkl"

    with pytest.raises(RuntimeError):
        with file_handling.open_atomic(target, "wb") as handle:
            pickle.dump({"plddt": [1.0]}, handle)
            raise RuntimeError("killed")

    assert list(tmp_path.iterdir()) == []


def test_open_atomic_writes_binary_files(tmp_path):
    target = tmp_path / "result_model.pkl"

    with file_handling.open_atomic(target, "wb") as handle:
        pickle.dump({"plddt": [1.0, 2.0]}, handle, protocol=4)

    with open(target, "rb") as handle:
        assert pickle.load(handle) == {"plddt": [1.0, 2.0]}


def test_open_atomic_files_follow_the_umask_rather_than_owner_only(tmp_path):
    previous = os.umask(0o022)
    try:
        with file_handling.open_atomic(tmp_path / "ranked_0.pdb") as handle:
            handle.write("model")
    finally:
        os.umask(previous)

    assert (tmp_path / "ranked_0.pdb").stat().st_mode & 0o777 == 0o644


def test_open_atomic_rejects_partial_writes(tmp_path):
    with pytest.raises(ValueError, match="whole files"):
        with file_handling.open_atomic(tmp_path / "timings.json", "a"):
            pass
