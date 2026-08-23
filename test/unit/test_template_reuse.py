"""Recovering an alignment to re-run template search over existing features."""

from types import SimpleNamespace

import numpy as np
import pytest

from alphapulldown.utils import template_reuse
from alphapulldown.utils.template_reuse import (
    SOURCE_RECONSTRUCTED,
    SOURCE_UNIREF90_FILE,
    msa_for_template_search,
    search_templates,
    stockholm_from_a3m,
    stockholm_from_feature_dict,
)

# Shaped like real jackhmmer output, including the #=GC RF annotation that
# AF2's column-pruning step needs to delimit the alignment chunk.
GOOD_STO = (
    "# STOCKHOLM 1.0\n\n"
    "query    ACDE\n"
    "hit1     ACDF\n"
    "#=GC RF  xxxx\n"
    "//\n"
)


# ------------------------------------------------------- Stockholm assembly

def test_stockholm_from_a3m_emits_a_parseable_alignment():
    sto = stockholm_from_a3m(">query\nACDE\n>hit1\nACDF\n")
    assert sto.startswith("# STOCKHOLM 1.0")
    assert sto.rstrip().endswith("//")
    rows = [
        ln.split() for ln in sto.splitlines()
        if ln and not ln.startswith("#") and ln.strip() != "//"
    ]
    assert rows == [["query", "ACDE"], ["hit1", "ACDF"]]


def test_stockholm_from_a3m_drops_lowercase_insertions():
    """A3M lowercase marks insertions relative to the query.

    They must go, otherwise rows are ragged and hmmbuild has no alignment.
    """
    sto = stockholm_from_a3m(">query\nACDE\n>hit1\nACfgDE\n")
    seqs = [
        ln.split()[1] for ln in sto.splitlines()
        if ln and not ln.startswith("#") and ln.strip() != "//"
    ]
    assert seqs == ["ACDE", "ACDE"]


def test_stockholm_from_a3m_disambiguates_repeated_names():
    """Stockholm keys rows by name, so duplicates would silently collapse."""
    sto = stockholm_from_a3m(">dup\nACDE\n>dup\nACDF\n>dup\nACDG\n")
    names = [
        ln.split()[0] for ln in sto.splitlines()
        if ln and not ln.startswith("#") and ln.strip() != "//"
    ]
    assert len(names) == len(set(names)) == 3


def test_stockholm_from_a3m_uses_only_the_first_token_of_a_header():
    sto = stockholm_from_a3m(">sp|P12345|NAME some description here\nACDE\n")
    names = [
        ln.split()[0] for ln in sto.splitlines()
        if ln and not ln.startswith("#") and ln.strip() != "//"
    ]
    assert names == ["sp|P12345|NAME"]


def test_stockholm_from_a3m_rejects_ragged_rows():
    with pytest.raises(ValueError, match="differ in length"):
        stockholm_from_a3m(">query\nACDE\n>hit1\nACD\n")


def test_stockholm_from_a3m_rejects_empty_input():
    with pytest.raises(ValueError, match="no sequences"):
        stockholm_from_a3m("")


# --------------------------------------------- reconstruction from features

def test_stockholm_from_feature_dict_decodes_the_integer_msa():
    # HHblits alphabet is alphabetical: 0=A, 1=C, 2=D, 3=E
    features = {"msa": np.asarray([[0, 1, 2, 3], [0, 1, 2, 2]], dtype=np.int32)}
    sto = stockholm_from_feature_dict(features)
    seqs = [
        ln.split()[1] for ln in sto.splitlines()
        if ln and not ln.startswith("#") and ln.strip() != "//"
    ]
    assert seqs == ["ACDE", "ACDD"]


@pytest.mark.parametrize(
    "features",
    [{}, {"msa": np.zeros((0, 0), dtype=np.int32)}, {"msa": np.zeros(4)}],
)
def test_stockholm_from_feature_dict_rejects_unusable_msas(features):
    with pytest.raises(ValueError):
        stockholm_from_feature_dict(features)


# ------------------------------------------------------------ source choice

def _monomer_with_msa():
    return SimpleNamespace(
        sequence="ACDE",
        feature_dict={"msa": np.asarray([[0, 1, 2, 3]], dtype=np.int32)},
    )


def test_msa_for_template_search_prefers_the_uniref90_file(tmp_path):
    """AF2 builds its template profile from uniref90 alone, so prefer that file."""
    (tmp_path / "uniref90_hits.sto").write_text(GOOD_STO)

    result = msa_for_template_search(_monomer_with_msa(), tmp_path)

    assert result.source == SOURCE_UNIREF90_FILE
    assert not result.is_reconstructed
    assert result.stockholm == GOOD_STO


def test_msa_for_template_search_falls_back_when_the_file_is_absent(tmp_path):
    """--save_msa_files=False deletes alignments once features are written."""
    result = msa_for_template_search(_monomer_with_msa(), tmp_path)

    assert result.source == SOURCE_RECONSTRUCTED
    assert result.is_reconstructed
    assert "ACDE" in result.stockholm


def test_msa_for_template_search_falls_back_when_the_file_is_unusable(tmp_path):
    """An empty uniref90_hits.sto must not be preferred over reconstruction."""
    (tmp_path / "uniref90_hits.sto").touch()

    result = msa_for_template_search(_monomer_with_msa(), tmp_path)

    assert result.source == SOURCE_RECONSTRUCTED


# --------------------------------------------------------------- the search

class _FakeSearcher:
    def __init__(self, input_format="sto"):
        self.input_format = input_format
        self.output_format = "sto"
        self.queried_with = None

    def query(self, text):
        self.queried_with = text
        return "RAW_HITS"

    def get_template_hits(self, output_string, input_sequence):
        assert output_string == "RAW_HITS"
        return ["hit-a", "hit-b"]


class _FakeFeaturizer:
    def __init__(self):
        self.seen_hits = None

    def get_templates(self, query_sequence, hits):
        self.seen_hits = hits
        return SimpleNamespace(
            features={"template_domain_names": np.asarray([b"1abc_A"], dtype=object)}
        )


def test_search_templates_runs_the_searcher_and_featurizer(tmp_path):
    searcher, featurizer = _FakeSearcher(), _FakeFeaturizer()

    features = search_templates(
        searcher, featurizer,
        query_sequence="ACDE",
        stockholm_msa=GOOD_STO,
        msa_output_dir=tmp_path,
    )

    assert features["template_domain_names"].tolist() == [b"1abc_A"]
    assert featurizer.seen_hits == ["hit-a", "hit-b"]
    # The hit file is written where AF2 would write it, for later inspection.
    assert (tmp_path / "pdb_hits.sto").read_text() == "RAW_HITS"


def test_search_templates_converts_to_a3m_for_an_a3m_searcher():
    """HHsearch consumes A3M; hmmsearch consumes Stockholm. Honour both."""
    searcher = _FakeSearcher(input_format="a3m")

    search_templates(
        searcher, _FakeFeaturizer(),
        query_sequence="ACDE",
        stockholm_msa=GOOD_STO,
        msa_output_dir=None,
    )

    assert searcher.queried_with.startswith(">")


def test_search_templates_rejects_an_unknown_input_format():
    searcher = _FakeSearcher(input_format="clustal")
    with pytest.raises(ValueError, match="Unrecognized template input format"):
        search_templates(
            searcher, _FakeFeaturizer(),
            query_sequence="ACDE",
            stockholm_msa=GOOD_STO,
            msa_output_dir=None,
        )


def test_search_templates_rejects_a_stockholm_without_reference_annotation():
    """Without '#=GC RF' the AF2 helper fails with a bare KeyError; be explicit."""
    with pytest.raises(ValueError, match="#=GC RF"):
        search_templates(
            _FakeSearcher(), _FakeFeaturizer(),
            query_sequence="ACDE",
            stockholm_msa="# STOCKHOLM 1.0\nquery ACDE\n//\n",
            msa_output_dir=None,
        )


def test_generated_stockholm_survives_the_af2_column_pruning():
    """The generator's output must be consumable by the real AF2 helper."""
    sto = stockholm_from_a3m(">query\nACDE\n>hit1\nACDF\n")
    features = search_templates(
        _FakeSearcher(), _FakeFeaturizer(),
        query_sequence="ACDE",
        stockholm_msa=sto,
        msa_output_dir=None,
    )
    assert "template_domain_names" in features


def test_search_templates_can_skip_writing_hits():
    searcher = _FakeSearcher()
    features = search_templates(
        searcher, _FakeFeaturizer(),
        query_sequence="ACDE",
        stockholm_msa=GOOD_STO,
        msa_output_dir=None,
    )
    assert "template_domain_names" in features
