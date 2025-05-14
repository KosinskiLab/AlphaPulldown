import pytest
from pathlib import Path
from alphapulldown.utils.modelling_setup import parse_fold

# Helper to write a minimal FASTA
def write_fasta(tmp_path: Path, name: str, sequences: list[str]) -> str:
    """
    Write one or more sequences to a FASTA file under tmp_path.
    `sequences` is a list of strings, each the raw sequence.
    Returns the full path as a string.
    """
    path = tmp_path / name
    # write each sequence with a distinct header
    lines = []
    for idx, seq in enumerate(sequences):
        lines.append(f">seq{idx}")
        lines.append(seq)
    path.write_text("\n".join(lines))
    return str(path)


def test_parse_all_region(tmp_path):
    fasta = write_fasta(tmp_path, "protein.fasta", ["ACDEFG"])
    result = parse_fold([f"{fasta}:0:all"])
    assert result == [[(fasta, 0, 1, 6)]]


def test_parse_numeric_range(tmp_path):
    fasta = write_fasta(tmp_path, "protein.fasta", ["MNOPQRST"])
    result = parse_fold([f"{fasta}:0:3-5"])
    assert result == [[(fasta, 0, 3, 5)]]


def test_parse_multiple_entries(tmp_path):
    # two chains in one job, plus a second job
    multi = write_fasta(tmp_path, "multi.fasta", ["AAAA", "BBBB"])
    jobs = [
        f"{multi}:1:2-3+{multi}:0:1-1",
        f"{multi}:0:all"
    ]
    result = parse_fold(jobs)
    assert result == [
        [(multi, 1, 2, 3), (multi, 0, 1, 1)],
        [(multi, 0, 1, 4)]
    ]


def test_bad_format_too_few_fields():
    with pytest.raises(ValueError):
        parse_fold(["badformat"])


def test_bad_region_format(tmp_path):
    fasta = write_fasta(tmp_path, "p.fasta", ["AAAAAA"])
    # 'x-y' is non-numeric
    with pytest.raises(ValueError):
        parse_fold([f"{fasta}:0:x-y"])


def test_missing_fasta_file():
    with pytest.raises(FileNotFoundError):
        parse_fold(["nofile.fasta:0:all"])


def test_chain_index_out_of_bounds(tmp_path):
    fasta = write_fasta(tmp_path, "p.fasta", ["AAAA"])
    # chain 1 does not exist
    with pytest.raises(IndexError):
        parse_fold([f"{fasta}:1:all"])
