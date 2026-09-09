"""Comparing MSA backends needs set overlap, not a ratio of counts.

The shipped comparison divided sequence counts and called it "recall", which is how
two entries in the published table read 101.3% and 103.2%. A ratio can exceed 1;
recall cannot. Worse, a ratio says nothing about whether the two backends found the
SAME sequences.
"""

from __future__ import annotations

from alphapulldown.utils.msa_quality import compare_a3m, neff


def _a3m(*sequences):
    """Header carries the accession, since that is what identifies a hit."""
    return "".join(f">{s}_acc\n{s}\n" for s in sequences)


def test_identical_counts_but_different_sequences_score_zero_recall():
    """The exact blind spot of a depth ratio."""
    reference = _a3m("QUERY", "AAAAA", "CCCCC")
    candidate = _a3m("QUERY", "DDDDD", "EEEEE")

    result = compare_a3m(reference, candidate)

    assert result["depth_ratio"] == 1.0     # counts match perfectly
    assert result["recall"] == 0.0          # nothing in common
    assert result["jaccard"] == 0.0


def test_recall_cannot_exceed_one_even_when_the_candidate_finds_more():
    """A depth ratio here would read 200%; recall is bounded by definition."""
    reference = _a3m("QUERY", "AAAAA")
    candidate = _a3m("QUERY", "AAAAA", "CCCCC")

    result = compare_a3m(reference, candidate)

    assert result["depth_ratio"] == 2.0
    assert result["recall"] == 1.0
    assert result["precision"] == 0.5


def test_alignment_differences_do_not_hide_a_shared_sequence():
    """Backends gap and lowercase against their own query; the homolog is the same."""
    reference = ">query\nQUERY\n>UniRef90_A d\nAC-DEF\n"
    candidate = ">query\nQUERY\n>UniRef90_A d\nACqDEF\n"

    assert compare_a3m(reference, candidate)["recall"] == 1.0


def test_the_query_row_is_not_counted_as_a_hit():
    reference = _a3m("QUERY", "AAAAA")

    assert compare_a3m(reference, reference)["reference_unique"] == 1


def test_an_empty_alignment_does_not_divide_by_zero():
    result = compare_a3m("", "")

    assert result["recall"] == 0.0 and result["jaccard"] == 0.0


def test_neff_discounts_redundancy_that_raw_depth_rewards():
    """Ten near-identical sequences are not worth ten independent ones."""
    redundant = _a3m("AAAAAAAAAA", *(["AAAAAAAAAA"] * 9))
    diverse = _a3m("AAAAAAAAAA", "CCCCCCCCCC", "DDDDDDDDDD", "EEEEEEEEEE")

    assert neff(redundant) < neff(diverse)
    assert neff(redundant) < 10


def test_the_same_homolog_aligned_differently_still_matches():
    """The bug this metric shipped with, pinned.

    jackhmmer writes `UniRef90_X/4-106 [subseq from] ...` and MMseqs2 writes
    `UniRef90_X ...`, and the two align the hit over slightly different extents. Keyed
    on residue string, identical alignments scored ~15% overlap; keyed on accession
    they score what they are.
    """
    reference = ">query\nAAAA\n>UniRef90_X/4-106 [subseq from] p\n-PLSV\n"
    candidate = ">query\nAAAA\n>UniRef90_X p\n--LSV\n"

    assert compare_a3m(reference, candidate)["recall"] == 1.0


def test_a_range_suffix_is_stripped_but_a_real_identifier_is_not():
    from alphapulldown.utils.msa_quality import _accession

    assert _accession("UniRef90_X/4-106 [subseq from] d") == "UniRef90_X"
    assert _accession("sp|P12345|NAME_HUMAN d") == "sp|P12345|NAME_HUMAN"
    assert _accession("UniRef90_X d") == "UniRef90_X"
