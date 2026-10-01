"""Contact extraction from AlphaFold 2 distograms.

``get_contacts`` used to reference a module global ``datadir`` that never existed,
so every call raised ``NameError``; and after choosing the top-ranked pickle it
re-read whichever file the scan had visited LAST.
"""

import pickle

import numpy as np

from alphapulldown.utils.distogram_parser import distogram_parser


def _payload(ranking_confidence, *, contact):
    logits = np.full((4, 4, 3), -10.0, dtype=np.float32)
    if contact:
        logits[0, 2, 0] = 10.0
        logits[2, 0, 0] = 10.0
    return {
        "ranking_confidence": ranking_confidence,
        "seqs": ["AA", "BB"],
        "distogram": {
            "bin_edges": np.array([4.0, 8.0, 12.0], dtype=np.float32),
            "logits": logits,
        },
    }


def _write(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def test_get_contacts_returns_empty_list_when_no_pickles_exist(tmp_path):
    assert distogram_parser().get_contacts(str(tmp_path)) == []


def test_get_contacts_reads_the_directory_it_is_given(tmp_path):
    _write(tmp_path / "result_model.pkl", _payload(0.9, contact=True))

    contacts = distogram_parser().get_contacts(
        str(tmp_path), distance=9, pbtycutoff=0.5, cross_only=True
    )

    assert len(contacts) == 1
    assert contacts[0][0] == (1, "A")
    assert contacts[0][1] == (1, "B")
    assert contacts[0][2] > 0.99


def test_get_contacts_reads_the_top_ranked_model_not_the_last_file_scanned(tmp_path, capsys):
    _write(tmp_path / "result_model_1.pkl", _payload(0.9, contact=True))
    _write(tmp_path / "result_model_2.pkl", _payload(0.5, contact=False))

    contacts = distogram_parser().get_contacts(
        str(tmp_path), distance=9, pbtycutoff=0.5, verbose=True
    )

    assert len(contacts) == 1
    assert "Selected result_model_1.pkl with ranking confidence 0.90" in capsys.readouterr().out


def test_select_top_ranked_pickle_needs_a_positive_ranking_confidence(tmp_path):
    _write(tmp_path / "result_model_1.pkl", {"ranking_confidence": 0.0})
    _write(tmp_path / "result_model_2.pkl", {"seqs": []})

    assert distogram_parser.select_top_ranked_pickle(str(tmp_path)) == (None, 0.0)
