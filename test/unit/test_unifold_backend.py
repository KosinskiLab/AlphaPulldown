"""The UniFold backend must be drivable through the prediction adapters.

Its ``setup()`` used to require ``output_dir`` and ``multimeric_object`` and its
``predict()`` took one object as an instance method, so neither adapter could
call it: ``--fold_backend=unifold`` failed on the first ``setup(**model_flags)``.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "folding_backend"
    / "unifold_backend.py"
)
MODULE_NAME = "alphapulldown.folding_backend.unifold_backend"


def _restore_modules(saved_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_unifold_stubs() -> dict[str, types.ModuleType | None]:
    names_to_replace = ["unifold", "unifold.config", "unifold.inference", "unifold.dataset"]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    unifold_pkg = types.ModuleType("unifold")
    unifold_pkg.__path__ = []  # type: ignore[attr-defined]
    config_mod = types.ModuleType("unifold.config")
    config_mod.model_config = lambda model_name: {"model_name": model_name}

    inference_mod = types.ModuleType("unifold.inference")
    inference_mod.calls = []
    inference_mod.config_args = (
        lambda model_dir, target_name, output_dir: {
            "model_dir": model_dir,
            "target_name": target_name,
            "output_dir": output_dir,
        }
    )
    inference_mod.unifold_config_model = lambda general_args: {"runner_args": general_args}
    inference_mod.unifold_predict = (
        lambda model_runner, model_args, processed_features: inference_mod.calls.append(
            (model_runner, model_args, processed_features)
        )
    )

    dataset_mod = types.ModuleType("unifold.dataset")
    dataset_mod.process_ap = (
        lambda config, features, mode, labels, seed, batch_idx, data_idx, is_distillation: (
            {"processed_features": features, "seed": seed, "mode": mode, "config": config},
            None,
        )
    )

    modules = {
        "unifold": unifold_pkg,
        "unifold.config": config_mod,
        "unifold.inference": inference_mod,
        "unifold.dataset": dataset_mod,
    }
    for name, module in modules.items():
        sys.modules[name] = module
    unifold_pkg.config = config_mod
    unifold_pkg.inference = inference_mod
    unifold_pkg.dataset = dataset_mod
    return saved_modules


@pytest.fixture
def unifold_backend_module():
    saved_modules = _install_unifold_stubs()
    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)
        _restore_modules(saved_modules)


def _unifold_flags(**overrides):
    defaults = dict(
        fold_backend="unifold", unifold_model_name="multimer_ft", num_cycle=3,
        data_directory="/weights", num_predictions_per_model=1, crosslinks=None,
        desired_num_res=None, desired_num_msa=None, skip_templates=False,
        allow_resume=True, num_diffusion_samples=5, num_recycles=10,
        save_embeddings=False, save_distogram=False,
        flash_attention_implementation="triton", buckets=["256"],
        jax_compilation_cache_dir=None, features_directory=["/features"],
        num_seeds=None, debug_templates=False, debug_msas=False, dropout=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_unifold_backend_runs_through_the_prediction_adapters(unifold_backend_module, tmp_path):
    from alphapulldown.folding_backend import FoldingBackendManager
    from alphapulldown.prediction.inference_flags import model_flags
    from alphapulldown.prediction.prediction_batch import (
        PredictionBatch,
        PredictionJob,
        PreparedPredictionAdapter,
    )

    fold = SimpleNamespace(description="A_and_B", feature_dict={"msa": [1, 2]})
    output_dir = tmp_path / "A_and_B"  # created by the backend
    adapter = PreparedPredictionAdapter(
        backend=FoldingBackendManager(),
        fold_backend="unifold",
        objects_to_model=[{"object": fold, "output_dir": str(output_dir)}],
        model_flags=model_flags(_unifold_flags()),
        postprocess_flags={},
        random_seed=7,
    )

    summary = PredictionBatch((PredictionJob("legacy", "", output_dir),)).run(adapter)

    assert summary.failures == ()
    assert summary.completed_job_ids == ("legacy",)
    assert output_dir.is_dir()
    general_args = {
        "model_dir": "/weights",
        "target_name": "A_and_B",
        "output_dir": str(output_dir),
    }
    assert sys.modules["unifold.inference"].calls == [
        (
            {"runner_args": general_args},
            general_args,
            {
                "processed_features": {"msa": [1, 2]},
                "seed": 7,
                "mode": "predict",
                "config": {"model_name": "multimer_ft"},
            },
        )
    ]


def test_unifold_predict_handles_every_object_in_order(unifold_backend_module, tmp_path):
    backend = unifold_backend_module.UnifoldBackend
    session = backend.setup(model_dir="/weights", model_name="multimer_af2")
    objects = [
        {"object": SimpleNamespace(description=name, feature_dict={"n": index}),
         "output_dir": str(tmp_path / name)}
        for index, name in enumerate(("first", "second"))
    ]

    records = list(backend.predict(objects, random_seed=3, model_dir="/weights", **session))

    assert [record["object"].description for record in records] == ["first", "second"]
    assert [record["output_dir"] for record in records] == [o["output_dir"] for o in objects]
    assert all(record["prediction_results"] == {} for record in records)
    targets = [call[1]["target_name"] for call in sys.modules["unifold.inference"].calls]
    assert targets == ["first", "second"]
    assert backend.postprocess(prediction_results={}, output_dir=str(tmp_path)) is None


def test_unifold_setup_rejects_an_alphafold_model_name(unifold_backend_module):
    with pytest.raises(ValueError, match="multimer_af2"):
        unifold_backend_module.UnifoldBackend.setup(model_dir="/weights", model_name="monomer_ptm")
