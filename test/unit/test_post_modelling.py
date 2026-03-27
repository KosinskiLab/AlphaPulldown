import gzip
import json
import pickle
import builtins
from pathlib import Path

import pytest

from alphapulldown.utils import post_modelling


def test_compress_file_gzips_and_removes_original(tmp_path):
    file_path = tmp_path / "result.pkl"
    file_path.write_bytes(b"payload")

    gz_path = post_modelling.compress_file(str(file_path))

    assert gz_path == str(file_path) + ".gz"
    assert not file_path.exists()
    with gzip.open(gz_path, "rb") as handle:
        assert handle.read() == b"payload"


def test_compress_result_pickles_only_compresses_pickle_files(tmp_path):
    pickle_path = tmp_path / "result.pkl"
    text_path = tmp_path / "notes.txt"
    pickle_path.write_bytes(b"payload")
    text_path.write_text("keep", encoding="utf-8")

    post_modelling.compress_result_pickles(str(tmp_path))

    assert not pickle_path.exists()
    assert (tmp_path / "result.pkl.gz").is_file()
    assert text_path.is_file()


def test_remove_keys_from_pickle_updates_pickle_in_place(tmp_path):
    file_path = tmp_path / "result.pkl"
    with open(file_path, "wb") as handle:
        pickle.dump({"keep": 1, "distogram": 2, "masked_msa": 3}, handle)

    post_modelling.remove_keys_from_pickle(str(file_path), ["distogram", "missing"])

    with open(file_path, "rb") as handle:
        payload = pickle.load(handle)

    assert payload == {"keep": 1, "masked_msa": 3}


def test_remove_irrelevant_pickles_keeps_only_best_pickle(tmp_path):
    best = tmp_path / "result_model_1.pkl"
    other = tmp_path / "result_model_2.pkl"
    note = tmp_path / "note.txt"
    best.write_bytes(b"best")
    other.write_bytes(b"other")
    note.write_text("keep", encoding="utf-8")

    post_modelling.remove_irrelevant_pickles(str(tmp_path), best.name)

    assert best.is_file()
    assert not other.exists()
    assert note.is_file()


def test_post_prediction_process_remove_keys_then_compress_and_prune(tmp_path):
    ranking_debug = tmp_path / "ranking_debug.json"
    ranking_debug.write_text(json.dumps({"order": ["model_1"]}), encoding="utf-8")
    result_best = tmp_path / "result_model_1.pkl"
    result_other = tmp_path / "result_model_2.pkl"
    with open(result_best, "wb") as handle:
        pickle.dump({"keep": 1, "distogram": 2, "masked_msa": 3}, handle)
    with open(result_other, "wb") as handle:
        pickle.dump({"keep": 4, "aligned_confidence_probs": 5}, handle)

    post_modelling.post_prediction_process(
        str(tmp_path),
        compress_pickles=True,
        remove_pickles=True,
        remove_keys=True,
    )

    assert not result_best.exists()
    assert not result_other.exists()
    gz_best = tmp_path / "result_model_1.pkl.gz"
    assert gz_best.is_file()
    with gzip.open(gz_best, "rb") as handle:
        payload = pickle.load(handle)
    assert payload == {"keep": 1}


def test_post_prediction_process_handles_missing_ranking_debug(tmp_path):
    lonely_pickle = tmp_path / "result_model_1.pkl"
    lonely_pickle.write_bytes(b"payload")

    post_modelling.post_prediction_process(str(tmp_path), compress_pickles=True)

    assert lonely_pickle.is_file()


def test_compress_file_returns_gz_path_even_when_open_fails(monkeypatch, tmp_path):
    file_path = tmp_path / "broken.pkl"
    file_path.write_bytes(b"payload")

    def fake_open(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(builtins, "open", fake_open)

    gz_path = post_modelling.compress_file(str(file_path))

    assert gz_path == str(file_path) + ".gz"
    assert file_path.is_file()
    assert not (tmp_path / "broken.pkl.gz").exists()


def test_remove_keys_from_pickle_logs_and_leaves_missing_pickle_untouched(tmp_path):
    missing = tmp_path / "missing.pkl"

    post_modelling.remove_keys_from_pickle(str(missing), ["distogram"])

    assert not missing.exists()


def test_post_prediction_process_compresses_all_pickles_without_removal(tmp_path):
    (tmp_path / "ranking_debug.json").write_text(json.dumps({"order": ["model_1"]}), encoding="utf-8")
    for name in ("result_model_1.pkl", "result_model_2.pkl"):
        with open(tmp_path / name, "wb") as handle:
            pickle.dump({"keep": name}, handle)

    post_modelling.post_prediction_process(str(tmp_path), compress_pickles=True, remove_pickles=False)

    assert not any(path.suffix == ".pkl" for path in tmp_path.iterdir())
    assert sorted(path.name for path in tmp_path.glob("*.gz")) == [
        "result_model_1.pkl.gz",
        "result_model_2.pkl.gz",
    ]


def test_post_prediction_process_removes_non_best_pickles_without_compressing(tmp_path):
    (tmp_path / "ranking_debug.json").write_text(json.dumps({"order": ["model_2"]}), encoding="utf-8")
    for name in ("result_model_1.pkl", "result_model_2.pkl", "result_model_3.pkl"):
        with open(tmp_path / name, "wb") as handle:
            pickle.dump({"keep": name}, handle)

    post_modelling.post_prediction_process(str(tmp_path), compress_pickles=False, remove_pickles=True)

    assert sorted(path.name for path in tmp_path.glob("*.pkl")) == ["result_model_2.pkl"]
    assert not list(tmp_path.glob("*.gz"))
