import json

import pytest

from alphapulldown.scripts.compare_msa_backends import compare_directories
from alphapulldown.utils.msa_quality import measure_a3m


def test_measure_a3m_reports_depth_uniqueness_and_query_coverage():
    metrics = measure_a3m(
        ">query\nACDE\n>full\nACDE\n>partial\nA--E\n>duplicate\nA--E\n",
        query_length=4,
    )

    assert metrics == {
        "depth": 4,
        "unique_depth": 2,
        "mean_non_gap_coverage": 0.75,
    }


def _feature(path, *, sequence="ACDE", unpaired=">query\nACDE\n"):
    path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "sequence": sequence,
                            "unpairedMsa": unpaired,
                            "pairedMsa": ">query\nACDE\n",
                            "templates": [],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_compare_directories_pairs_every_artifact_and_reports_depth(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _feature(reference / "alpha_af3_input.json")
    _feature(
        candidate / "alpha_af3_input.json",
        unpaired=">query\nACDE\n>hit\nAC-E\n",
    )

    rows = compare_directories(reference, candidate)

    assert len(rows) == 1
    assert rows[0]["reference_unpaired"]["depth"] == 1
    assert rows[0]["candidate_unpaired"]["depth"] == 2


def test_compare_directories_rejects_unpaired_artifact_sets(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _feature(reference / "missing_af3_input.json")

    with pytest.raises(ValueError, match="missing from candidate"):
        compare_directories(reference, candidate)


def _af2_pickle(path, sequence, msa_rows, *, deletions=None, species=(), templates=()):
    """An AlphaFold 2 feature pickle, as far as the comparison reads one.

    Pickles the lightweight stand-in, so this needs no AlphaFold import.
    """
    import pickle

    import numpy as np

    from alphapulldown.utils.af2_to_af3_msa import AF2_ID_TO_A3M
    from alphapulldown.utils.lightweight_pickles import LightweightMonomericObject

    msa = np.array([[AF2_ID_TO_A3M.index(r) for r in row] for row in msa_rows],
                   dtype=np.int32)
    feature_dict = {
        "msa": msa,
        "deletion_matrix_int": (
            np.asarray(deletions) if deletions is not None else np.zeros_like(msa)
        ),
        "msa_all_seq": msa[: 1 + len(species)],
        "msa_species_identifiers_all_seq": np.array(
            [b"", *[s.encode() for s in species]], dtype=object
        ),
        "template_domain_names": np.array(
            [t.encode() for t in templates] or [b""], dtype=object
        ),
    }
    monomer = LightweightMonomericObject(
        description=path.stem, sequence=sequence, feature_dict=feature_dict
    )
    path.write_bytes(pickle.dumps(monomer))


def test_af2_comparison_measures_what_each_pickle_gives_the_model(tmp_path):
    from alphapulldown.scripts.compare_msa_backends import (
        compare_af2_directories,
        summarize_af2,
    )

    reference, candidate = tmp_path / "native", tmp_path / "local"
    reference.mkdir()
    candidate.mkdir()
    _af2_pickle(reference / "alpha.pkl", "ACDEFG", ["ACDEFG", "ACDEFW", "ACDEWG"],
                deletions=[[0] * 6, [0, 2, 0, 0, 0, 0], [0] * 6],
                species=("HUMAN", "MOUSE"), templates=("1abc_A",))
    _af2_pickle(candidate / "alpha.pkl", "ACDEFG", ["ACDEFG", "ACDEFW"],
                species=("HUMAN",))

    [row] = compare_af2_directories(reference, candidate)

    assert row["reference"]["unpaired"]["depth"] == 3
    assert row["candidate"]["unpaired"]["depth"] == 2
    assert row["reference"]["rows_with_insertions"] == 1
    assert row["candidate"]["rows_with_insertions"] == 0
    assert row["reference"]["paired_species"] == 2
    assert row["candidate"]["paired_species"] == 1
    assert row["reference"]["template_count"] == 1
    # AlphaFold 2's empty-template placeholder is not a template.
    assert row["candidate"]["template_count"] == 0
    assert summarize_af2([row])["mean_candidate_paired_species"] == 1


def test_af2_comparison_refuses_unpaired_or_mismatched_sets(tmp_path):
    from alphapulldown.scripts.compare_msa_backends import compare_af2_directories

    reference, candidate = tmp_path / "native", tmp_path / "local"
    reference.mkdir()
    candidate.mkdir()
    _af2_pickle(reference / "alpha.pkl", "ACDEFG", ["ACDEFG"])
    _af2_pickle(candidate / "beta.pkl", "ACDEFG", ["ACDEFG"])
    with pytest.raises(ValueError, match="differ"):
        compare_af2_directories(reference, candidate)

    (candidate / "beta.pkl").unlink()
    _af2_pickle(candidate / "alpha.pkl", "ACDEFW", ["ACDEFW"])
    with pytest.raises(ValueError, match="Sequence mismatch"):
        compare_af2_directories(reference, candidate)
