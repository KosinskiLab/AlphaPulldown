import pickle

import numpy as np

import alphapulldown.utils.distogram_parser as distogram_parser_module


def test_get_contacts_returns_empty_list_when_no_pickles_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(distogram_parser_module, "datadir", str(tmp_path), raising=False)

    parser = distogram_parser_module.distogram_parser()

    assert parser.get_contacts("ignored") == []


def test_get_contacts_extracts_top_ranked_inter_chain_contact(monkeypatch, tmp_path):
    logits = np.full((4, 4, 3), -10.0, dtype=np.float32)
    logits[0, 2, 0] = 10.0
    logits[2, 0, 0] = 10.0
    payload = {
        "ranking_confidence": 0.9,
        "seqs": ["AA", "BB"],
        "distogram": {
            "bin_edges": np.array([4.0, 8.0, 12.0], dtype=np.float32),
            "logits": logits,
        },
    }
    with open(tmp_path / "result_model.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    monkeypatch.setattr(distogram_parser_module, "datadir", str(tmp_path), raising=False)

    parser = distogram_parser_module.distogram_parser()
    contacts = parser.get_contacts("ignored", distance=9, pbtycutoff=0.5, cross_only=True)

    assert len(contacts) == 1
    assert contacts[0][0] == (1, "A")
    assert contacts[0][1] == (1, "B")
    assert contacts[0][2] > 0.99
