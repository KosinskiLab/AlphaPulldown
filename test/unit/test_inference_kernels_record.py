"""The record of which kernels predicted a fold, in its alphapulldown/ folder."""

import json

from alphapulldown.prediction import inference_kernels


def _record(output_dir):
    return json.loads((output_dir / "alphapulldown" / "inference_kernels.json").read_text())


def test_entries_accumulate_in_the_alphapulldown_folder(tmp_path):
    inference_kernels.record_inference_kernels(tmp_path, "seed-1", {"fused": True})
    inference_kernels.record_inference_kernels(tmp_path, "seed-2", {"fused": False})

    assert _record(tmp_path) == {"seed-1": {"fused": True}, "seed-2": {"fused": False}}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["alphapulldown"]


def test_a_record_left_by_2_10_is_moved_into_the_folder(tmp_path):
    # Resuming a fold predicted by 2.10, which kept the record at the top.
    (tmp_path / "inference_kernels.json").write_text(json.dumps({"model_1": {"fused": False}}))

    assert inference_kernels.read_inference_kernels(tmp_path) == {"model_1": {"fused": False}}
    inference_kernels.record_inference_kernels(tmp_path, "model_2", {"fused": True})

    assert _record(tmp_path) == {"model_1": {"fused": False}, "model_2": {"fused": True}}
    assert not (tmp_path / "inference_kernels.json").exists()


def test_an_unreadable_record_is_replaced(tmp_path, caplog):
    path = tmp_path / "alphapulldown" / "inference_kernels.json"
    path.parent.mkdir()
    path.write_text("{truncated")

    inference_kernels.record_inference_kernels(tmp_path, "seed-1", {"fused": True})

    assert _record(tmp_path) == {"seed-1": {"fused": True}}
    assert "Replacing unreadable" in caplog.text


def test_a_record_that_cannot_be_written_never_fails_the_prediction(tmp_path, caplog):
    # alphapulldown/ is a file, so the folder cannot be made.
    (tmp_path / "alphapulldown").write_text("in the way")

    inference_kernels.record_inference_kernels(tmp_path, "seed-1", {"fused": True})

    assert "Not recording the inference kernels of seed-1" in caplog.text
