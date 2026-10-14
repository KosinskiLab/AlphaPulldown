"""The --storage_mode presets for an AlphaFold 3 output directory."""

import csv
import json

from alphapulldown.folding_backend import alphafold3_storage


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
    alphafold3_storage.post_prediction_process_af3(str(tmp_path), job, storage_mode="vanilla")
    assert (tmp_path / f"{job}_confidences.json").is_file()
    assert (tmp_path / "seed-9_sample-0" / "confidences.json").is_file()


def test_af3_storage_mode_slim_keeps_best_plain_xz_others(tmp_path):
    job = _make_af3_dir(tmp_path)
    alphafold3_storage.post_prediction_process_af3(str(tmp_path), job, storage_mode="slim")
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
    alphafold3_storage.post_prediction_process_af3(str(tmp_path), job, storage_mode="minimal")
    assert (tmp_path / "seed-9_sample-1" / "confidences.json").is_file()  # best plain
    for s in ("0", "2"):
        sd = tmp_path / f"seed-9_sample-{s}"
        assert (sd / "confidences.json.xz").is_file()  # compressed, not deleted
        assert not (sd / "confidences.json").exists()
        assert (sd / "model.cif").is_file() and (sd / "summary_confidences.json").is_file()


def test_af3_storage_mode_preserves_top_confidences_when_best_sample_lacks_it(tmp_path):
    job = _make_af3_dir(tmp_path)
    (tmp_path / "seed-9_sample-1" / "confidences.json").unlink()
    alphafold3_storage.post_prediction_process_af3(str(tmp_path), job, storage_mode="slim")
    # top-level copy is the only remaining source of best-model PAE -> must survive
    assert (tmp_path / f"{job}_confidences.json").is_file()


def test_af3_best_sample_needs_a_readable_ranking(tmp_path):
    assert alphafold3_storage.af3_best_sample_dir(tmp_path) is None
    # Truncated by a killed run of a version before atomic writes: no scores.
    (tmp_path / "ranking_scores.csv").write_text("seed,sample\n9,0\n", encoding="utf-8")
    assert alphafold3_storage.af3_ranking_scores(tmp_path) == []
    assert alphafold3_storage.af3_best_sample_dir(tmp_path) is None


def test_af3_storage_mode_never_fails_a_finished_fold(tmp_path, caplog):
    alphafold3_storage.post_prediction_process_af3(str(tmp_path / "gone"), "job", storage_mode="slim")
    assert "AF3 post-processing error" in caplog.text

    # A rerun after an earlier slim pass already removed one duplicate.
    job = _make_af3_dir(tmp_path)
    (tmp_path / f"{job}_data.json").unlink()
    alphafold3_storage.post_prediction_process_af3(str(tmp_path), job, storage_mode="slim")
    assert not (tmp_path / f"{job}_confidences.json").exists()
    assert (tmp_path / "seed-9_sample-1" / "confidences.json").is_file()
