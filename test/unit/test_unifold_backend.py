import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "folding_backend"
    / "unifold_backend.py"
)


def _restore_modules(saved_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_unifold_backend_stubs() -> dict[str, types.ModuleType | None]:
    names_to_replace = [
        "alphapulldown.objects",
        "unifold",
        "unifold.config",
        "unifold.inference",
        "unifold.dataset",
    ]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    objects_mod = types.ModuleType("alphapulldown.objects")
    objects_mod.MultimericObject = type("MultimericObject", (), {})

    unifold_pkg = types.ModuleType("unifold")
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
            {
                "processed_features": features,
                "seed": seed,
                "mode": mode,
                "config": config,
            },
            None,
        )
    )

    modules = {
        "alphapulldown.objects": objects_mod,
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


def _load_unifold_backend_module():
    saved_modules = _install_unifold_backend_stubs()
    sys.modules.pop("alphapulldown.folding_backend.unifold_backend", None)
    spec = importlib.util.spec_from_file_location(
        "alphapulldown.folding_backend.unifold_backend",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module, saved_modules
    except Exception:
        sys.modules.pop(spec.name, None)
        _restore_modules(saved_modules)
        raise


def test_unifold_setup_predict_and_postprocess():
    module, saved_modules = _load_unifold_backend_module()
    try:
        multimeric_object = SimpleNamespace(description="complex", feature_dict={"msa": [1, 2]})

        configured = module.UnifoldBackend.setup(
            model_name="multimer",
            model_dir="/models",
            output_dir="/output",
            multimeric_object=multimeric_object,
        )
        assert configured == {
            "model_runner": {
                "runner_args": {
                    "model_dir": "/models",
                    "target_name": "complex",
                    "output_dir": "/output",
                }
            },
            "model_args": {
                "model_dir": "/models",
                "target_name": "complex",
                "output_dir": "/output",
            },
            "model_config": {"model_name": "multimer"},
        }

        backend = module.UnifoldBackend()
        assert (
            backend.predict(
                model_runner="runner",
                model_args={"arg": 1},
                model_config={"cfg": 2},
                multimeric_object=multimeric_object,
                random_seed=11,
            )
            is None
        )

        inference_mod = sys.modules["unifold.inference"]
        assert inference_mod.calls == [
            (
                "runner",
                {"arg": 1},
                {
                    "processed_features": {"msa": [1, 2]},
                    "seed": 11,
                    "mode": "predict",
                    "config": {"cfg": 2},
                },
            )
        ]
        assert module.UnifoldBackend.postprocess() is None
    finally:
        sys.modules.pop("alphapulldown.folding_backend.unifold_backend", None)
        _restore_modules(saved_modules)
