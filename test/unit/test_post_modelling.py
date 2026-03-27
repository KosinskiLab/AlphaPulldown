import gzip
import json
import pickle
from pathlib import Path

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
