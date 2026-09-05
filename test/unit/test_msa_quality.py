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
