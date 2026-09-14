"""Cutting an MSA bundle into the alignments AlphaFold 2 builds features from.

No AlphaFold import: this is string slicing, and it has to be right in every
environment -- a wrong cut does not fail, it hands AlphaFold 2 another database's
rows or a template profile built from the wrong alignment.
"""

from __future__ import annotations

import pytest

from alphapulldown import af2_feature_finalizer
from alphapulldown.af2_feature_finalizer import af2_msa_inputs
from alphapulldown.feature_batch import SearchedMsas


QUERY = "MKTAYI"


def _msas(uniref90=2, mgnify=2, small_bfd=2, paired=3) -> SearchedMsas:
    """Distinguishable rows: U/M/B for each database, P for UniProt."""

    def rows(prefix, count):
        return [(f"{prefix}{index}", f"{prefix}KTAYI") for index in range(count)]

    unpaired = [("query", QUERY), *rows("U", uniref90), *rows("M", mgnify)]
    unpaired += rows("B", small_bfd)
    return SearchedMsas(
        unpaired="".join(f">{d}\n{s}\n" for d, s in unpaired),
        paired="".join(
            f">{d}\n{s}\n" for d, s in [("query", QUERY), *rows("P", paired)]
        ),
        unpaired_rows=(
            ("uniref90", uniref90),
            ("mgnify", mgnify),
            ("small_bfd", small_bfd),
        ),
    )


def _names(a3m: str) -> list[str]:
    return [line[1:] for line in a3m.splitlines() if line.startswith(">")]


def test_main_alignment_follows_alphafold2_merge_order():
    # The bundle merges uniref90, mgnify, small_bfd; AlphaFold 2 merges uniref90,
    # BFD, MGnify (pipeline.py:265). Order decides what its MSA sampling sees first.
    inputs = af2_msa_inputs(_msas())
    assert _names(inputs.main_a3m) == ["query", "U0", "U1", "B0", "B1", "M0", "M1"]


def test_templates_come_from_uniref90_alone():
    inputs = af2_msa_inputs(_msas())
    assert _names(inputs.template_a3m) == ["query", "U0", "U1"]


def test_caps_count_the_query_row_as_alphafold2_does(monkeypatch):
    # jackhmmer's max_sto_sequences stops at that many sequence NAMES, query first,
    # so a cap of 3 means the query plus two hits.
    monkeypatch.setattr(
        af2_feature_finalizer, "AF2_MAX_SEQUENCES", {"uniref90": 3, "mgnify": 2}
    )
    inputs = af2_msa_inputs(_msas(uniref90=5, mgnify=5, small_bfd=5))

    assert _names(inputs.template_a3m) == ["query", "U0", "U1"]
    assert _names(inputs.main_a3m) == [
        "query", "U0", "U1", "B0", "B1", "B2", "B3", "B4", "M0",
    ]


def test_paired_alignment_is_uniprot_truncated_like_all_seq_msa_features(monkeypatch):
    monkeypatch.setattr(af2_feature_finalizer, "PAIRED_MAX_SEQUENCES", 3)
    inputs = af2_msa_inputs(_msas(paired=10))
    assert _names(inputs.paired_a3m) == ["query", "P0", "P1"]


def test_real_defaults_are_alphafold2s():
    assert af2_feature_finalizer.AF2_MAX_SEQUENCES == {
        "uniref90": 10_000,
        "mgnify": 501,
    }
    assert af2_feature_finalizer.PAIRED_MAX_SEQUENCES == 50_000


def test_a_bundle_without_row_spans_is_refused():
    """Without spans there is no uniref90 to build templates from -- and no way to
    fake one, since the merged alignment has lost the boundaries."""
    msas = _msas()
    spanless = SearchedMsas(unpaired=msas.unpaired, paired=msas.paired)
    with pytest.raises(ValueError):
        af2_msa_inputs(spanless)


def test_a_bundle_with_no_paired_alignment_is_refused():
    msas = _msas()
    unpaired_only = SearchedMsas(
        unpaired=msas.unpaired, paired="", unpaired_rows=msas.unpaired_rows
    )
    with pytest.raises(ValueError, match="paired"):
        af2_msa_inputs(unpaired_only)
