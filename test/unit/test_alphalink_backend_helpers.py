import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "folding_backend"
    / "alphalink_backend.py"
)


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _restore_modules(saved_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _tensor_tree_map(func, tree):
    if isinstance(tree, dict):
        return {key: _tensor_tree_map(func, value) for key, value in tree.items()}
    return func(tree)


def _install_alphalink_backend_stubs() -> dict[str, types.ModuleType | None]:
    names_to_replace = [
        "alphapulldown.folding_backend.alphafold2_backend",
        "alphapulldown.objects",
        "alphapulldown.utils.plotting",
        "torch",
        "unifold",
        "unifold.config",
        "unifold.modules",
        "unifold.modules.alphafold",
        "unifold.dataset",
        "unifold.data",
        "unifold.data.residue_constants",
        "unifold.data.protein",
        "unifold.data.data_ops",
        "unicore",
        "unicore.utils",
    ]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    af2_backend_mod = types.ModuleType("alphapulldown.folding_backend.alphafold2_backend")
    af2_backend_mod._save_pae_json_file = (
        lambda pae, max_pae, output_dir, model_name: Path(output_dir, f"pae_{model_name}.json").write_text(
            json.dumps({"max_pae": max_pae}),
            encoding="utf-8",
        )
    )

    objects_mod = types.ModuleType("alphapulldown.objects")

    class MonomericObject:
        pass

    class MultimericObject:
        pass

    class ChoppedObject:
        pass

    objects_mod.MonomericObject = MonomericObject
    objects_mod.MultimericObject = MultimericObject
    objects_mod.ChoppedObject = ChoppedObject

    plotting_mod = types.ModuleType("alphapulldown.utils.plotting")
    plotting_mod.plot_pae_from_matrix = lambda *args, **kwargs: None

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.half = "half"

    class FakeTensor:
        def __init__(self, value, dtype=None):
            self._value = np.asarray(value)
            self.dtype = dtype if dtype is not None else self._value.dtype

        def float(self):
            return FakeTensor(self._value.astype(np.float32), np.float32)

        def cpu(self):
            return self._value

    torch_mod.FakeTensor = FakeTensor
    torch_mod.cuda = SimpleNamespace(
        is_available=lambda: False,
        current_device=lambda: 0,
        get_device_properties=lambda _device: SimpleNamespace(
            total_memory=40 * 1024 * 1024 * 1024
        ),
    )
    torch_mod.load = lambda path: {
        "ema": {"params": {"module.layer": 1, "module.bias": 2}}
    }
    torch_mod.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, exc_type, exc, tb: False,
    )
    torch_mod.autograd = SimpleNamespace(set_detect_anomaly=lambda flag: None)
    torch_mod.as_tensor = lambda value, device=None: value
    torch_mod.from_numpy = lambda value: value

    unifold_pkg = _package("unifold")
    config_mod = types.ModuleType("unifold.config")

    def _model_config(_name):
        return SimpleNamespace(
            data=SimpleNamespace(
                common=SimpleNamespace(max_recycling_iters=None),
                predict=SimpleNamespace(
                    num_ensembles=None,
                    subsample_templates=False,
                ),
            ),
            globals=SimpleNamespace(max_recycling_iters=None),
        )

    config_mod.model_config = _model_config

    modules_pkg = _package("unifold.modules")
    alphafold_mod = types.ModuleType("unifold.modules.alphafold")

    class FakeAlphaFold:
        def __init__(self, config):
            self.config = config
            self.loaded_state_dict = None
            self.device = None
            self.eval_called = False
            self.inference_mode_called = False
            self.bfloat16_called = False
            self.globals = SimpleNamespace(chunk_size=None, block_size=None)

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True

        def inference_mode(self):
            self.inference_mode_called = True

        def bfloat16(self):
            self.bfloat16_called = True

    alphafold_mod.AlphaFold = FakeAlphaFold

    dataset_mod = types.ModuleType("unifold.dataset")
    dataset_mod.process_ap = lambda **kwargs: ({}, None)

    data_pkg = _package("unifold.data")
    residue_constants_mod = types.ModuleType("unifold.data.residue_constants")
    residue_constants_mod.atom_order = {"CA": 1}
    residue_constants_mod.atom_type_num = 37
    protein_mod = types.ModuleType("unifold.data.protein")
    protein_mod.from_prediction = lambda **kwargs: SimpleNamespace(chain_index=np.array([[0]]), aatype=np.array([[0]]))
    protein_mod.to_pdb = lambda protein: "PDB"
    data_ops_mod = types.ModuleType("unifold.data.data_ops")
    data_ops_mod.get_pairwise_distances = lambda coords: np.zeros((1, 1))

    unicore_pkg = _package("unicore")
    unicore_utils_mod = types.ModuleType("unicore.utils")
    unicore_utils_mod.tensor_tree_map = _tensor_tree_map

    modules = {
        "alphapulldown.folding_backend.alphafold2_backend": af2_backend_mod,
        "alphapulldown.objects": objects_mod,
        "alphapulldown.utils.plotting": plotting_mod,
        "torch": torch_mod,
        "unifold": unifold_pkg,
        "unifold.config": config_mod,
        "unifold.modules": modules_pkg,
        "unifold.modules.alphafold": alphafold_mod,
        "unifold.dataset": dataset_mod,
        "unifold.data": data_pkg,
        "unifold.data.residue_constants": residue_constants_mod,
        "unifold.data.protein": protein_mod,
        "unifold.data.data_ops": data_ops_mod,
        "unicore": unicore_pkg,
        "unicore.utils": unicore_utils_mod,
    }

    for name, module in modules.items():
        sys.modules[name] = module

    unifold_pkg.config = config_mod
    unifold_pkg.modules = modules_pkg
    unifold_pkg.dataset = dataset_mod
    unifold_pkg.data = data_pkg
    modules_pkg.alphafold = alphafold_mod
    data_pkg.residue_constants = residue_constants_mod
    data_pkg.protein = protein_mod
    data_pkg.data_ops = data_ops_mod
    unicore_pkg.utils = unicore_utils_mod

    return saved_modules


@pytest.fixture(scope="module")
def alphalink_backend_module():
    saved_modules = _install_alphalink_backend_stubs()
    sys.modules.pop("alphapulldown.folding_backend.alphalink_backend", None)
    spec = importlib.util.spec_from_file_location(
        "alphapulldown.folding_backend.alphalink_backend",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)
        _restore_modules(saved_modules)


def test_setup_accepts_expected_weights_and_validates_missing_inputs(
    alphalink_backend_module,
    monkeypatch,
    tmp_path,
):
    warning_messages = []
    monkeypatch.setattr(alphalink_backend_module.logging, "warning", warning_messages.append)

    direct_file = tmp_path / "custom_weights.pt"
    direct_file.write_text("stub", encoding="utf-8")
    direct_setup = alphalink_backend_module.AlphaLinkBackend.setup(str(direct_file))
    assert direct_setup["param_path"] == str(direct_file)
    assert warning_messages

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    canonical_file = weights_dir / "AlphaLink-Multimer_SDA_v3.pt"
    canonical_file.write_text("stub", encoding="utf-8")
    dir_setup = alphalink_backend_module.AlphaLinkBackend.setup(str(weights_dir))
    assert dir_setup["param_path"] == str(canonical_file)

    wrong_extension = tmp_path / "weights.bin"
    wrong_extension.write_text("stub", encoding="utf-8")
    with pytest.raises(ValueError, match=".pt extension"):
        alphalink_backend_module.AlphaLinkBackend.setup(str(wrong_extension))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        alphalink_backend_module.AlphaLinkBackend.setup(str(tmp_path / "missing"))


def test_unload_tensors_and_prepare_model_runner(alphalink_backend_module):
    fake_tensor = alphalink_backend_module.torch.FakeTensor
    batch, out = alphalink_backend_module.AlphaLinkBackend.unload_tensors(
        {"a": fake_tensor([1, 2], dtype=alphalink_backend_module.torch.bfloat16)},
        {"b": fake_tensor([3, 4], dtype=np.float32)},
    )
    np.testing.assert_array_equal(batch["a"], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(out["b"], np.array([3, 4], dtype=np.float32))

    model = alphalink_backend_module.AlphaLinkBackend.prepare_model_runner(
        "weights.pt",
        bf16=True,
        model_device="cpu",
    )
    assert model.device == "cpu"
    assert model.eval_called is True
    assert model.inference_mode_called is True
    assert model.bfloat16_called is True
    assert model.loaded_state_dict == {"layer": 1, "bias": 2}
    assert (
        model.config.data.common.max_recycling_iters
        == alphalink_backend_module.MAX_RECYCLING_ITERS
    )
    assert (
        model.config.data.predict.num_ensembles
        == alphalink_backend_module.NUM_ENSEMBLES
    )


def test_resume_preprocess_and_chunk_helpers(alphalink_backend_module, tmp_path):
    (tmp_path / "AlphaLink2_model_0_seed_123_0.875.pdb").write_text("pdb", encoding="utf-8")
    (tmp_path / "pae_AlphaLink2_model_0_seed_123_0.875.json").write_text(
        "{}",
        encoding="utf-8",
    )
    already_exists, iptm_value = alphalink_backend_module.AlphaLinkBackend.check_resume_status(
        "AlphaLink2_model_0_seed_123",
        str(tmp_path),
    )
    assert already_exists is True
    assert iptm_value == pytest.approx(0.875)

    processed = alphalink_backend_module.AlphaLinkBackend.preprocess_features(
        {
            "seq_length": np.array([7, 7]),
            "num_alignments": np.array([3, 3]),
            "num_templates": np.array([2, 2]),
            "template_all_atom_masks": np.ones((1, 7, 37), dtype=np.float32),
            "template_aatype": np.eye(22)[[0, 1, 2, 3, 4, 5, 6]][None, :, :],
            "template_sum_probs": np.array([0.5], dtype=np.float32),
            "deletion_matrix_int": np.ones((1, 7), dtype=np.float32),
            "deletion_matrix_int_all_seq": np.full((2, 7), 2.0, dtype=np.float32),
            "msa": np.ones((2, 7), dtype=np.int32),
        }
    )
    assert processed["seq_length"] == 7
    assert processed["num_alignments"] == 3
    assert processed["num_templates"] == 2
    assert processed["template_all_atom_mask"].shape == (1, 7, 37)
    assert processed["template_aatype"].shape == (1, 7)
    assert processed["template_sum_probs"].shape == (1, 1)
    np.testing.assert_array_equal(processed["deletion_matrix"], np.ones((1, 7)))
    np.testing.assert_array_equal(
        processed["extra_deletion_matrix"],
        np.full((2, 7), 2.0),
    )
    assert processed["msa_mask"].shape == (2, 7)
    assert processed["msa_row_mask"].shape == (2,)
    np.testing.assert_array_equal(processed["asym_id"], np.zeros(7, dtype=np.int32))
    np.testing.assert_array_equal(processed["entity_id"], np.zeros(7, dtype=np.int32))
    np.testing.assert_array_equal(processed["sym_id"], np.ones(7, dtype=np.int32))

    assert alphalink_backend_module.AlphaLinkBackend.automatic_chunk_size(500, "cpu") == (
        256,
        None,
    )
    assert alphalink_backend_module.AlphaLinkBackend.automatic_chunk_size(1000, "cpu") == (
        128,
        None,
    )
    assert alphalink_backend_module.AlphaLinkBackend.automatic_chunk_size(1500, "cpu") == (
        64,
        None,
    )
    assert alphalink_backend_module.AlphaLinkBackend.automatic_chunk_size(2200, "cpu") == (
        32,
        512,
    )
    assert alphalink_backend_module.AlphaLinkBackend.automatic_chunk_size(3000, "cpu") == (
        4,
        256,
    )


def test_predict_validates_inputs_and_builds_chain_maps(
    alphalink_backend_module,
    monkeypatch,
    tmp_path,
):
    with pytest.raises(ValueError, match="Missing required parameters"):
        list(alphalink_backend_module.AlphaLinkBackend.predict(objects_to_model=[]))

    captured_calls = []

    def fake_predict_iterations(feature_dict, output_dir, **kwargs):
        captured_calls.append((feature_dict, output_dir, kwargs))

    monkeypatch.setattr(
        alphalink_backend_module.AlphaLinkBackend,
        "predict_iterations",
        staticmethod(fake_predict_iterations),
    )

    default_chain_object = SimpleNamespace(
        feature_dict={"x": 1},
        input_seqs=["AAA"],
    )
    integer_chain_map_object = SimpleNamespace(
        feature_dict={"y": 2},
        input_seqs=["AAA", "BBB"],
        chain_id_map={"A": 0, "B": 1},
    )

    results = list(
        alphalink_backend_module.AlphaLinkBackend.predict(
            objects_to_model=[
                {"object": default_chain_object, "output_dir": str(tmp_path / "default")},
                {"object": integer_chain_map_object, "output_dir": str(tmp_path / "int_map")},
            ],
            configs=SimpleNamespace(data="cfg"),
            param_path="weights.pt",
            num_predictions_per_model=2,
        )
    )

    assert len(results) == 2
    assert captured_calls[0][2]["crosslinks"] == ""
    default_chain_map = captured_calls[0][2]["chain_id_map"]
    assert sorted(default_chain_map) == ["A"]
    assert default_chain_map["A"].description == "default_chain"
    assert default_chain_map["A"].sequence == "AAA"

    integer_chain_map = captured_calls[1][2]["chain_id_map"]
    assert integer_chain_map["A"].description == "chain_A"
    assert integer_chain_map["A"].sequence == "AAA"
    assert integer_chain_map["B"].description == "chain_B"
    assert integer_chain_map["B"].sequence == "BBB"
    assert captured_calls[1][2]["num_inference"] == 2


def test_postprocess_ranks_nested_prediction_outputs(alphalink_backend_module, tmp_path):
    low_dir = tmp_path / "seed0"
    high_dir = tmp_path / "seed1"
    low_dir.mkdir()
    high_dir.mkdir()
    (low_dir / "AlphaLink2_model_0_seed_1_0.500.pdb").write_text("LOW", encoding="utf-8")
    (high_dir / "AlphaLink2_model_1_seed_2_0.900.pdb").write_text("HIGH", encoding="utf-8")

    alphalink_backend_module.AlphaLinkBackend.postprocess({}, str(tmp_path))

    ranking = json.loads((tmp_path / "ranking_debug.json").read_text(encoding="utf-8"))
    assert ranking["order"] == [
        "AlphaLink2_model_1_seed_2_0.900",
        "AlphaLink2_model_0_seed_1_0.500",
    ]
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "HIGH"
    assert (tmp_path / "ranked_1.pdb").read_text(encoding="utf-8") == "LOW"
