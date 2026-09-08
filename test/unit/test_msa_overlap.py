"""Comparing MSA backends needs set overlap, not a ratio of counts.

The shipped comparison divided sequence counts and called it "recall", which is how
two entries in the published table read 101.3% and 103.2%. A ratio can exceed 1;
recall cannot. Worse, a ratio says nothing about whether the two backends found the
SAME sequences.
"""

from __future__ import annotations

from alphapulldown.utils.msa_quality import compare_a3m, neff


def _a3m(*sequences):
    return "".join(f">s{i}\n{s}\n" for i, s in enumerate(sequences))


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
    reference = _a3m("QUERY", "AC-DEF")
    candidate = _a3m("QUERY", "ACqDEF")

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
