import csv
import gzip
import json
import lzma
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


# --- storage_mode presets (AF2) ---

def _make_af2_dir(tmp_path):
    order = ["model_1", "model_2"]
    (tmp_path / "ranking_debug.json").write_text(json.dumps({"order": order}), encoding="utf-8")
    for m in order:
        with open(tmp_path / f"result_{m}.pkl", "wb") as handle:
            pickle.dump(
                {"predicted_aligned_error": [[0.0]], "max_predicted_aligned_error": 30.0,
                 "iptm": 0.8, "plddt": [1.0], "distogram": 1, "masked_msa": 2},
                handle,
            )
        (tmp_path / f"pae_{m}.json").write_text("[]", encoding="utf-8")
    return order


def test_storage_mode_vanilla_leaves_pickles_untouched(tmp_path):
    _make_af2_dir(tmp_path)
    post_modelling.post_prediction_process(str(tmp_path), storage_mode="vanilla")
    pkls = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert pkls == ["result_model_1.pkl", "result_model_2.pkl"]
    payload = pickle.load(open(tmp_path / "result_model_1.pkl", "rb"))
    assert "predicted_aligned_error" in payload  # vanilla keeps everything


def test_storage_mode_slim_strips_pae_and_xz_compresses(tmp_path):
    order = _make_af2_dir(tmp_path)
    post_modelling.post_prediction_process(str(tmp_path), storage_mode="slim")
    assert not list(tmp_path.glob("*.pkl"))  # all compressed
    xz = sorted(tmp_path.glob("*.pkl.xz"))
    assert len(xz) == 2
    payload = pickle.loads(lzma.open(xz[0]).read())
    assert "predicted_aligned_error" not in payload  # PAE stripped (kept in sidecar)
    assert "max_predicted_aligned_error" not in payload
    assert "distogram" not in payload and "masked_msa" not in payload
    assert "iptm" in payload and "plddt" in payload  # scores retained
    for m in order:  # pae sidecars (what AlphaJudge reads) survive
        assert (tmp_path / f"pae_{m}.json").is_file()


def test_storage_mode_minimal_deletes_all_pickles_keeps_sidecars(tmp_path):
    order = _make_af2_dir(tmp_path)
    post_modelling.post_prediction_process(str(tmp_path), storage_mode="minimal")
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pkl.xz"))
    assert (tmp_path / "ranking_debug.json").is_file()
    for m in order:
        assert (tmp_path / f"pae_{m}.json").is_file()


# --- storage_mode presets (AF3) ---

def _make_af3_dir(tmp_path, job="A_and_B"):
    big = json.dumps({"pae": [[0.0]]})
    (tmp_path / f"{job}_confidences.json").write_text(big, encoding="utf-8")
    (tmp_path / f"{job}_data.json").write_text("{}", encoding="utf-8")
    (tmp_path / f"{job}_model.cif").write_text("data_x\n", encoding="utf-8")
    (tmp_path / f"{job}_summary_confidences.json").write_text("{}", encoding="utf-8")
    rows = [("9", "0", 0.5), ("9", "1", 0.9), ("9", "2", 0.3)]  # best = sample-1
    with open(tmp_path / "ranking_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "sample", "ranking_score"])
        w.writerows(rows)
    for seed, sample, score in rows:
        sd = tmp_path / f"seed-{seed}_sample-{sample}"
        sd.mkdir()
        (sd / "confidences.json").write_text(big, encoding="utf-8")
        (sd / "model.cif").write_text("data_x\n", encoding="utf-8")
        (sd / "summary_confidences.json").write_text(json.dumps({"ranking_score": score}), encoding="utf-8")
    return job


def test_af3_storage_mode_vanilla_is_noop(tmp_path):
    job = _make_af3_dir(tmp_path)
    post_modelling.post_prediction_process_af3(str(tmp_path), job, storage_mode="vanilla")
    assert (tmp_path / f"{job}_confidences.json").is_file()
    assert (tmp_path / "seed-9_sample-0" / "confidences.json").is_file()


def test_af3_storage_mode_slim_keeps_best_plain_xz_others(tmp_path):
    job = _make_af3_dir(tmp_path)
    post_modelling.post_prediction_process_af3(str(tmp_path), job, storage_mode="slim")
    # best (sample-1) plain for AlphaJudge; others xz
    assert (tmp_path / "seed-9_sample-1" / "confidences.json").is_file()
    assert not (tmp_path / "seed-9_sample-1" / "confidences.json.xz").exists()
    for s in ("0", "2"):
        assert (tmp_path / f"seed-9_sample-{s}" / "confidences.json.xz").is_file()
        assert not (tmp_path / f"seed-9_sample-{s}" / "confidences.json").exists()
    # top-level duplicates removed, structure kept
    assert not (tmp_path / f"{job}_confidences.json").exists()
    assert not (tmp_path / f"{job}_data.json").exists()
    assert (tmp_path / f"{job}_model.cif").is_file()


def test_af3_storage_mode_minimal_matches_slim_compresses_non_best(tmp_path):
    # AF3 minimal == slim: non-best confidences are xz-compressed, never deleted,
    # so the full per-sample PAE matrix is never lost (deleting it would silently
    # degrade AlphaJudge to the coarser summary_confidences.json PAE).
    job = _make_af3_dir(tmp_path)
    post_modelling.post_prediction_process_af3(str(tmp_path), job, storage_mode="minimal")
    assert (tmp_path / "seed-9_sample-1" / "confidences.json").is_file()  # best plain
    for s in ("0", "2"):
        sd = tmp_path / f"seed-9_sample-{s}"
        assert (sd / "confidences.json.xz").is_file()  # compressed, not deleted
        assert not (sd / "confidences.json").exists()
        assert (sd / "model.cif").is_file() and (sd / "summary_confidences.json").is_file()


def test_af3_storage_mode_preserves_top_confidences_when_best_sample_lacks_it(tmp_path):
    job = _make_af3_dir(tmp_path)
    (tmp_path / "seed-9_sample-1" / "confidences.json").unlink()
    post_modelling.post_prediction_process_af3(str(tmp_path), job, storage_mode="slim")
    # top-level copy is the only remaining source of best-model PAE -> must survive
    assert (tmp_path / f"{job}_confidences.json").is_file()
