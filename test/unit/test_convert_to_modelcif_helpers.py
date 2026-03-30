import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("ihm")
pytest.importorskip("modelcif")

import alphapulldown.scripts.convert_to_modelcif as convert_to_modelcif


TEST_PREDICTIONS_DIR = (
    Path(__file__).resolve().parents[1] / "test_data" / "predictions"
)


def test_cast_param_preserves_expected_python_types():
    assert convert_to_modelcif._cast_param("7") == 7
    assert convert_to_modelcif._cast_param("3.5") == 3.5
    assert convert_to_modelcif._cast_param("True") is True
    assert convert_to_modelcif._cast_param("False") is False
    assert convert_to_modelcif._cast_param("plain-text") == "plain-text"


def test_compress_cif_file_replaces_plaintext_file(tmp_path):
    cif_file = tmp_path / "ranked_0.cif"
    cif_file.write_text("data_test\n", encoding="ascii")

    compressed_path = convert_to_modelcif._compress_cif_file(str(cif_file))

    assert compressed_path.endswith(".gz")
    assert not cif_file.exists()
    assert (tmp_path / "ranked_0.cif.gz").exists()


def test_get_feature_metadata_falls_back_to_structure_sequence(tmp_path):
    source_dir = TEST_PREDICTIONS_DIR / "TEST"
    work_dir = tmp_path / "TEST"
    shutil.copytree(source_dir, work_dir)

    metadata_file = next(work_dir.glob("*_feature_metadata_*.json"))
    payload = json.loads(metadata_file.read_text(encoding="ascii"))
    payload["other"]["fasta_paths"] = "['/this/path/does/not/exist.fasta']"
    metadata_file.write_text(json.dumps(payload, indent=2), encoding="ascii")

    modelcif_json = {}
    complex_name, fasta_dicts = convert_to_modelcif._get_feature_metadata(
        modelcif_json,
        "TEST",
        str(work_dir),
        fallback_structure_path=str(work_dir / "ranked_0.pdb"),
    )

    assert complex_name == "TEST"
    assert fasta_dicts
    assert fasta_dicts[0]["description"] == "chain_A"
    assert fasta_dicts[0]["sequence"].startswith("MESAIA")
    assert modelcif_json["__meta__"]["TEST"]["databases"]


def test_get_model_list_selects_requested_model_and_tracks_non_selected_models():
    selected_models = convert_to_modelcif._get_model_list(
        str(TEST_PREDICTIONS_DIR / "TEST"),
        0,
        True,
    )

    assert len(selected_models) == 1
    result = selected_models[0]
    assert result["complex"] == "TEST"
    assert len(result["models"]) == 1
    assert result["models"][0][0].endswith("ranked_0.pdb")
    assert len(result["not_selected"]) == 4


def test_main_processes_associated_models_before_selected_models(monkeypatch, tmp_path):
    calls = []

    def fake_convert(complex_name, model_tuple, out_dir, compress, additional_assoc_files=None):
        calls.append(
            {
                "complex_name": complex_name,
                "model_tuple": model_tuple,
                "out_dir": out_dir,
                "compress": compress,
                "additional_assoc_files": additional_assoc_files,
            }
        )
        return {
            f"{Path(model_tuple[0]).stem}.zip": (
                str(Path(out_dir) / f"{Path(model_tuple[0]).stem}.zip"),
                object(),
            )
        }

    monkeypatch.setattr(
        convert_to_modelcif,
        "FLAGS",
        SimpleNamespace(
            ap_output=str(tmp_path / "predictions"),
            model_selected=0,
            add_associated=True,
            compress=False,
        ),
    )
    monkeypatch.setattr(
        convert_to_modelcif,
        "_get_model_list",
        lambda ap_output, model_selected, get_non_selected: [
            {
                "complex": "TEST",
                "path": str(tmp_path / "out"),
                "models": [("ranked_0.pdb", "result_0.pkl", "0", 0)],
                "not_selected": [("ranked_1.pdb", "result_1.pkl", "1", 1)],
            }
        ],
    )
    monkeypatch.setattr(
        convert_to_modelcif,
        "alphapulldown_model_to_modelcif",
        fake_convert,
    )

    convert_to_modelcif.main([])

    assert len(calls) == 2
    assert calls[0]["model_tuple"][0] == "ranked_1.pdb"
    assert calls[0]["additional_assoc_files"] is None
    assert calls[0]["out_dir"] != str(tmp_path / "out")
    assert calls[1]["model_tuple"][0] == "ranked_0.pdb"
    assert calls[1]["additional_assoc_files"]
