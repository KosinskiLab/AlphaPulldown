"""The unavailable legacy backend must fail before importing a model runtime."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from alphapulldown.folding_backend import FoldingBackendManager
from alphapulldown.folding_backend.unifold_backend import UnifoldBackend
from alphapulldown.prediction.prediction_batch import (
    PredictionBatch, PredictionJob, PreparedPredictionAdapter,
)


@pytest.mark.parametrize("model_name", [
    "multimer_af2", "multimer_ft", "multimer", "multimer_af2_v3",
    "multimer_af2_model45_v3",
])
def test_legacy_model_choices_fail_with_actionable_error(model_name, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "unifold", None)
    with pytest.raises(ValueError, match="UniFold.*unavailable.*AlphaLink"):
        UnifoldBackend.setup(model_dir="/missing/weights", model_name=model_name)


def test_direct_prediction_rejects_unifold_before_creating_outputs(tmp_path):
    output = tmp_path / "fold"
    with pytest.raises(ValueError, match="UniFold.*unavailable"):
        list(UnifoldBackend.predict(
            objects_to_model=[{"object": SimpleNamespace(), "output_dir": str(output)}],
            model_dir="/missing/weights", model_config={},
        ))
    assert not output.exists()


def test_prediction_adapter_reports_unavailable_backend(tmp_path):
    output = tmp_path / "fold"
    adapter = PreparedPredictionAdapter(
        backend=FoldingBackendManager(), fold_backend="unifold",
        objects_to_model=[{"object": SimpleNamespace(), "output_dir": str(output)}],
        model_flags={"model_dir": "/missing/weights", "model_name": "multimer_af2"},
        postprocess_flags={}, random_seed=7,
    )
    # Backend setup is a batch-level rejection, not a recoverable fold failure.
    with pytest.raises(ValueError, match="UniFold.*unavailable"):
        PredictionBatch((PredictionJob("legacy", "", Path(output)),)).run(adapter)
    assert not output.exists()


def test_unifold_is_not_advertised_as_available(monkeypatch):
    import alphapulldown.folding_backend as manager_module
    monkeypatch.setattr(manager_module, "_try_import", lambda *args: object)
    assert "unifold" not in FoldingBackendManager().available_backends()
