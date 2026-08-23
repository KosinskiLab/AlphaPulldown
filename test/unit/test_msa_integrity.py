"""Structural checks on precomputed MSA files.

The cases here are the ones actually observed in a large feature-generation
run: a zero-length Stockholm left by a tool that exited after creating its
output file, and alignments cut off mid-write when a job was killed.
"""

import gzip
import lzma

import pytest

from alphapulldown.utils.msa_integrity import (
    check_a3m,
    check_msa_file,
    check_stockholm,
    validate_precomputed_msas,
)

GOOD_STO = "# STOCKHOLM 1.0\n\nquery  ACDE\nhit1   ACDF\n//\n"
GOOD_A3M = ">query\nACDE\n>hit1\nACDF\n"


# --------------------------------------------------------------- Stockholm

@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        ("", "empty"),
        ("   \n\n", "empty"),
        ("query ACDE\n//\n", "missing '# STOCKHOLM' header"),
        # jackhmmer killed mid-write: header and rows present, no terminator.
        ("# STOCKHOLM 1.0\n\nquery  ACDE\nhit1   ACDF\n", "truncated"),
        ("# STOCKHOLM 1.0\n#=GF ID x\n//\n", "no alignment rows"),
    ],
)
def test_check_stockholm_rejects(text, expected_fragment):
    problem = check_stockholm(text)
    assert problem is not None
    assert expected_fragment in problem


def test_check_stockholm_accepts_a_complete_alignment():
    assert check_stockholm(GOOD_STO) is None


# --------------------------------------------------------------------- A3M

@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        ("", "empty"),
        ("ACDE\nACDF\n", "no '>' header lines"),
        # Killed just after writing a description line.
        (">query\nACDE\n>hit1\n", "truncated"),
    ],
)
def test_check_a3m_rejects(text, expected_fragment):
    problem = check_a3m(text)
    assert problem is not None
    assert expected_fragment in problem


def test_check_a3m_accepts_a_complete_alignment():
    assert check_a3m(GOOD_A3M) is None


# -------------------------------------------------------------- file level

def test_check_msa_file_flags_a_zero_byte_stockholm(tmp_path):
    """The exact failure seen in production: jackhmmer left an empty .sto.

    Reused, it surfaces as StopIteration inside deduplicate_stockholm_msa,
    which names neither the file nor the protein.
    """
    empty = tmp_path / "uniref90_hits.sto"
    empty.touch()
    assert "empty" in check_msa_file(empty)


def test_check_msa_file_ignores_non_alignment_files(tmp_path):
    """Search results and profiles are not alignments and must not be judged."""
    # pdb_hits.sto is hmmsearch *output* in Stockholm format: it looks like an
    # alignment, but an empty one just means no hits were found.
    for name in ("pdb_hits.hhr", "pdb_hits.sto", "query.hmm", "features.pkl"):
        path = tmp_path / name
        path.write_text("whatever")
        assert check_msa_file(path) is None


def test_check_msa_file_reports_a_missing_file(tmp_path):
    assert check_msa_file(tmp_path / "uniref90_hits.sto") == "file does not exist"


@pytest.mark.parametrize("wrapper", ["gz", "xz"])
def test_check_msa_file_reads_through_compression(tmp_path, wrapper):
    """zip_msa_files gzips alignments in place, so checks must see through it."""
    path = tmp_path / f"uniref90_hits.sto.{wrapper}"
    opener = gzip.open if wrapper == "gz" else lzma.open
    with opener(path, "wt") as handle:
        handle.write(GOOD_STO)
    assert check_msa_file(path) is None


def test_check_msa_file_flags_a_corrupt_compressed_alignment(tmp_path):
    path = tmp_path / "uniref90_hits.sto.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00 truncated garbage")
    assert "unreadable" in check_msa_file(path)


# ---------------------------------------------------------------- directory

def test_validate_precomputed_msas_reports_only_bad_files(tmp_path):
    (tmp_path / "uniref90_hits.sto").write_text(GOOD_STO)
    (tmp_path / "bfd_uniref_hits.a3m").write_text(">q\nACDE\n>h\n")  # truncated
    (tmp_path / "mgnify_hits.sto").touch()                           # empty
    (tmp_path / "pdb_hits.hhr").write_text("not an alignment")
    (tmp_path / "pdb_hits.sto").touch()   # a search that found nothing

    problems = validate_precomputed_msas(tmp_path)

    assert {p.path.name for p in problems} == {
        "bfd_uniref_hits.a3m", "mgnify_hits.sto",
    }
    # A sound alignment must survive an inspection that does not remove.
    assert (tmp_path / "uniref90_hits.sto").exists()


def test_validate_precomputed_msas_removes_only_the_bad_ones(tmp_path):
    good = tmp_path / "uniref90_hits.sto"
    good.write_text(GOOD_STO)
    bad = tmp_path / "mgnify_hits.sto"
    bad.touch()
    unrelated = tmp_path / "pdb_hits.hhr"
    unrelated.write_text("hits")

    problems = validate_precomputed_msas(tmp_path, remove_invalid=True)

    assert [p.path.name for p in problems] == ["mgnify_hits.sto"]
    assert not bad.exists(), "unusable alignment should be gone so it is regenerated"
    assert good.exists(), "sound alignment must be kept - regenerating it is expensive"
    assert unrelated.exists()


def test_validate_precomputed_msas_keeps_an_empty_template_hit_file(tmp_path):
    """An empty pdb_hits.sto means "no templates matched", not a broken file."""
    hits = tmp_path / "pdb_hits.sto"
    hits.touch()
    assert validate_precomputed_msas(tmp_path, remove_invalid=True) == []
    assert hits.exists()


def test_validate_precomputed_msas_tolerates_a_missing_directory(tmp_path):
    assert validate_precomputed_msas(tmp_path / "nope") == []


def test_validate_precomputed_msas_ignores_subdirectories(tmp_path):
    (tmp_path / "nested").mkdir()
    assert validate_precomputed_msas(tmp_path) == []
