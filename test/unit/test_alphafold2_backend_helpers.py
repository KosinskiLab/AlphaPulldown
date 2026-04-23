import importlib.util
import json
import pickle
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
    / "alphafold2_backend.py"
)


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


class _ConfigNode(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _restore_modules(saved_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_alphafold2_backend_stubs() -> dict[str, types.ModuleType | None]:
    names_to_replace = [
        "jax",
        "jax.numpy",
        "alphafold",
        "alphafold.relax",
        "alphafold.relax.relax",
        "alphafold.common",
        "alphafold.common.protein",
        "alphafold.common.residue_constants",
        "alphafold.common.confidence",
        "alphafold.model",
        "alphafold.model.config",
        "alphafold.model.data",
        "alphafold.model.model",
        "alphapulldown.objects",
        "alphapulldown.utils.plotting",
        "alphapulldown.utils.post_modelling",
        "alphapulldown.utils.modelling_setup",
        "alphapulldown.utils.af2_to_af3_msa",
    ]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    jax_pkg = _package("jax")
    jax_numpy_mod = types.ModuleType("jax.numpy")
    jax_numpy_mod.ndarray = np.ndarray
    jax_numpy_mod.array = np.array

    alphafold_pkg = _package("alphafold")
    relax_pkg = _package("alphafold.relax")
    relax_mod = types.ModuleType("alphafold.relax.relax")

    class FakeAmberRelaxation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process(self, prot):
            name = getattr(prot, "name", "protein")
            return (f"RELAXED:{name}", None, [0, 1])

    relax_mod.AmberRelaxation = FakeAmberRelaxation

    common_pkg = _package("alphafold.common")
    protein_mod = types.ModuleType("alphafold.common.protein")

    class FakeProtein:
        def __init__(
            self,
            *,
            atom_positions,
            atom_mask,
            aatype,
            residue_index,
            chain_index,
            b_factors,
        ):
            self.atom_positions = atom_positions
            self.atom_mask = atom_mask
            self.aatype = aatype
            self.residue_index = residue_index
            self.chain_index = chain_index
            self.b_factors = b_factors
            self.name = "debug_protein"

    def _to_pdb(prot):
        return f"PDB:{getattr(prot, 'name', 'protein')}"

    def _from_prediction(features, result, b_factors, remove_leading_feature_dimension):
        residue_index = np.asarray(features.get("residue_index", [0]), dtype=np.int32)
        chain_index = np.asarray(features.get("asym_id", np.zeros_like(residue_index)))
        return SimpleNamespace(
            name="predicted",
            features=features,
            result=result,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
            aatype=np.zeros_like(residue_index),
            remove_leading_feature_dimension=remove_leading_feature_dimension,
        )

    protein_mod.Protein = FakeProtein
    protein_mod.to_pdb = _to_pdb
    protein_mod.from_prediction = _from_prediction
    protein_mod.from_pdb_string = lambda text: SimpleNamespace(name=f"from_pdb:{text}")

    residue_constants_mod = types.ModuleType("alphafold.common.residue_constants")
    residue_constants_mod.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = list(range(64))
    residue_constants_mod.restype_num = 20
    residue_constants_mod.atom_type_num = 37

    confidence_mod = types.ModuleType("alphafold.common.confidence")
    confidence_mod.pae_json = lambda pae, max_pae: json.dumps(
        {"pae": np.asarray(pae).tolist(), "max_pae": float(max_pae)}
    )
    confidence_mod.confidence_json = lambda plddt: json.dumps(
        {"plddt": np.asarray(plddt).tolist()}
    )
    confidence_mod.predicted_tm_score = (
        lambda logits, breaks, asym_id=None, interface=False: 0.8 if interface else 0.5
    )
    confidence_mod.compute_predicted_aligned_error = lambda logits, breaks: {
        "predicted_aligned_error": np.asarray(logits) + 1,
        "max_predicted_aligned_error": 31.0,
    }

    model_pkg = _package("alphafold.model")
    config_mod = types.ModuleType("alphafold.model.config")
    config_mod.MODEL_PRESETS = {
        "multimer": ("model_1_multimer_v3", "model_2_multimer_v3")
    }

    def _model_config(_name):
        return _ConfigNode(
            {
                "model": _ConfigNode(
                    {
                        "num_ensemble_eval": None,
                        "global_config": _ConfigNode({"eval_dropout": False}),
                        "embeddings_and_evoformer": _ConfigNode(
                            {"num_msa": 64, "num_extra_msa": 256}
                        ),
                    }
                )
            }
        )

    config_mod.model_config = _model_config

    data_mod = types.ModuleType("alphafold.model.data")
    data_mod.get_model_haiku_params = (
        lambda model_name, data_dir: {"model_name": model_name, "data_dir": data_dir}
    )

    model_mod = types.ModuleType("alphafold.model.model")

    class FakeRunModel:
        def __init__(self, config, params):
            self.config = config
            self.params = params
            self.multimer_mode = True

        def process_features(self, feature_dict, random_seed):
            return dict(feature_dict)

        def predict(self, processed_feature_dict, random_seed):
            seq_len = len(np.asarray(processed_feature_dict.get("residue_index", [0, 1])))
            return {
                "plddt": np.full(seq_len, 75.0, dtype=np.float32),
                "predicted_aligned_error": np.zeros((seq_len, seq_len), dtype=np.float32),
                "max_predicted_aligned_error": 31.0,
            }

    model_mod.RunModel = FakeRunModel

    objects_mod = types.ModuleType("alphapulldown.objects")

    class MonomericObject:
        def __init__(self, description="monomer", sequence=""):
            self.description = description
            self.sequence = sequence
            self.feature_dict = {}
            self.multimeric_mode = False

    class MultimericObject:
        def __init__(
            self,
            description="multimer",
            input_seqs=None,
            feature_dict=None,
            multimeric_mode=True,
        ):
            self.description = description
            self.input_seqs = input_seqs or []
            self.feature_dict = feature_dict or {}
            self.multimeric_mode = multimeric_mode

    class ChoppedObject(MonomericObject):
        pass

    objects_mod.MonomericObject = MonomericObject
    objects_mod.MultimericObject = MultimericObject
    objects_mod.ChoppedObject = ChoppedObject

    plotting_mod = types.ModuleType("alphapulldown.utils.plotting")
    plotting_mod.plot_pae_from_matrix = lambda **_kwargs: None

    post_modelling_mod = types.ModuleType("alphapulldown.utils.post_modelling")
    post_modelling_mod.post_prediction_process = lambda *args, **kwargs: None

    modelling_setup_mod = types.ModuleType("alphapulldown.utils.modelling_setup")
    modelling_setup_mod.pad_input_features = (
        lambda feature_dict, desired_num_msa, desired_num_res: None
    )

    af2_to_af3_msa_mod = types.ModuleType("alphapulldown.utils.af2_to_af3_msa")
    af2_to_af3_msa_mod.msa_rows_and_deletions_to_a3m = (
        lambda msa_rows, deletion_rows, query_sequence: (
            f">query\n{query_sequence}\n>rows\n{len(np.asarray(msa_rows))}\n"
        )
    )

    modules = {
        "jax": jax_pkg,
        "jax.numpy": jax_numpy_mod,
        "alphafold": alphafold_pkg,
        "alphafold.relax": relax_pkg,
        "alphafold.relax.relax": relax_mod,
        "alphafold.common": common_pkg,
        "alphafold.common.protein": protein_mod,
        "alphafold.common.residue_constants": residue_constants_mod,
        "alphafold.common.confidence": confidence_mod,
        "alphafold.model": model_pkg,
        "alphafold.model.config": config_mod,
        "alphafold.model.data": data_mod,
        "alphafold.model.model": model_mod,
        "alphapulldown.objects": objects_mod,
        "alphapulldown.utils.plotting": plotting_mod,
        "alphapulldown.utils.post_modelling": post_modelling_mod,
        "alphapulldown.utils.modelling_setup": modelling_setup_mod,
        "alphapulldown.utils.af2_to_af3_msa": af2_to_af3_msa_mod,
    }

    for name, module in modules.items():
        sys.modules[name] = module

    jax_pkg.numpy = jax_numpy_mod
    alphafold_pkg.relax = relax_pkg
    alphafold_pkg.common = common_pkg
    alphafold_pkg.model = model_pkg
    relax_pkg.relax = relax_mod
    common_pkg.protein = protein_mod
    common_pkg.residue_constants = residue_constants_mod
    common_pkg.confidence = confidence_mod
    model_pkg.config = config_mod
    model_pkg.data = data_mod
    model_pkg.model = model_mod

    return saved_modules


@pytest.fixture(scope="module")
def af2_backend_module():
    saved_modules = _install_alphafold2_backend_stubs()
    sys.modules.pop("alphapulldown.folding_backend.alphafold2_backend", None)
    spec = importlib.util.spec_from_file_location(
        "alphapulldown.folding_backend.alphafold2_backend",
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


def test_json_template_and_alignment_helpers(af2_backend_module, tmp_path):
    missing = af2_backend_module._read_from_json_if_exists(tmp_path / "missing.json")
    assert missing == {}

    existing_path = tmp_path / "timings.json"
    existing_path.write_text(json.dumps({"stage": 1}), encoding="utf-8")
    assert af2_backend_module._read_from_json_if_exists(existing_path) == {"stage": 1}

    feature_dict = {
        "seq_length": 3,
        "template_aatype": np.ones((2, 3), dtype=np.int32),
        "template_all_atom_positions": np.ones((2, 3, 37, 3), dtype=np.float32),
        "template_all_atom_mask": np.zeros((2, 3, 37), dtype=np.float32),
        "num_templates": np.array([5]),
    }
    af2_backend_module._reset_template_features(feature_dict)
    assert feature_dict["template_aatype"].shape == (1, 3)
    assert feature_dict["template_all_atom_positions"].shape == (1, 3, 37, 3)
    assert np.all(feature_dict["template_all_atom_mask"] == 1.0)
    np.testing.assert_array_equal(feature_dict["num_templates"], np.array([1]))

    assert af2_backend_module._normalise_num_alignments_for_debug({}) == 0
    assert (
        af2_backend_module._normalise_num_alignments_for_debug({"msa": np.zeros((2, 3))})
        == 2
    )
    assert (
        af2_backend_module._normalise_num_alignments_for_debug(
            {"msa": np.zeros((2, 3)), "num_alignments": np.array([9])}
        )
        == 2
    )


def test_ensure_typing_dataclass_transform_backfills_missing_attribute(
    af2_backend_module, monkeypatch
):
    monkeypatch.delattr(af2_backend_module.typing, "dataclass_transform", raising=False)

    af2_backend_module._ensure_typing_dataclass_transform()

    assert af2_backend_module.typing.dataclass_transform is not None


def test_resolve_gpu_relax_keeps_cuda_when_available(af2_backend_module, monkeypatch):
    monkeypatch.setattr(
        af2_backend_module,
        "_get_openmm_platform_names",
        lambda: ["Reference", "CPU", "CUDA"],
    )

    assert af2_backend_module._resolve_gpu_relax(True) is True


def test_resolve_gpu_relax_falls_back_when_cuda_is_missing(
    af2_backend_module, monkeypatch, caplog
):
    monkeypatch.setattr(
        af2_backend_module,
        "_get_openmm_platform_names",
        lambda: ["Reference", "CPU", "OpenCL"],
    )

    assert af2_backend_module._resolve_gpu_relax(True) is False
    assert "falling back to CPU relax" in caplog.text


def test_asym_query_and_msa_debug_helpers(af2_backend_module, tmp_path):
    normalized = af2_backend_module._normalize_asym_id(
        {"asym_id": np.array([5, 5, 2, 9], dtype=np.int32)}
    )
    np.testing.assert_array_equal(normalized["asym_id"], np.array([0, 0, 1, 2]))

    fallback_normalized = af2_backend_module._normalize_asym_id(
        {},
        fallback_feature_dict={"asym_id": np.array([1, 1, 3], dtype=np.int32)},
    )
    np.testing.assert_array_equal(fallback_normalized["asym_id"], np.array([0, 0, 1]))

    multimer = af2_backend_module.MultimericObject(
        description="job",
        input_seqs=["AA", "BB"],
        feature_dict={},
    )
    monomer = af2_backend_module.MonomericObject("single", "AC")
    assert af2_backend_module._query_sequence_for_debug(multimer) == "AABB"
    assert af2_backend_module._query_sequence_for_debug(monomer) == "AC"

    af2_backend_module._write_processed_msa_debug_artifact(
        processed_feature_dict={
            "msa": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.zeros((2, 2), dtype=np.int32),
            "num_alignments": np.array([1]),
        },
        multimeric_object=monomer,
        output_dir=tmp_path,
        model_name="modelA",
    )

    debug_path = tmp_path / "modelA_processed_msa.a3m"
    assert debug_path.read_text(encoding="utf-8") == ">query\nAC\n>rows\n1\n"


def test_template_debug_helpers_cover_success_and_failure_paths(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    assert af2_backend_module._decode_debug_value(b"templ") == "templ"
    assert (
        af2_backend_module._sanitize_debug_filename("bad/name:*", "fallback")
        == "bad_name__"
    )

    one_hot = np.eye(22, dtype=np.int32)[[1, 2]]
    np.testing.assert_array_equal(
        af2_backend_module._template_aatype_to_indices(one_hot),
        np.array([1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        af2_backend_module._template_aatype_to_indices(np.array([0, 21], dtype=np.int32)),
        np.array([0, 20], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="Unsupported template_aatype rank"):
        af2_backend_module._template_aatype_to_indices(np.zeros((1, 1, 1, 1)))

    call_counter = {"value": 0}

    def fake_protein_constructor(**kwargs):
        current = call_counter["value"]
        call_counter["value"] += 1
        if current == 1:
            raise RuntimeError("boom")
        return SimpleNamespace(name="template", **kwargs)

    monkeypatch.setattr(af2_backend_module.protein, "Protein", fake_protein_constructor)
    monkeypatch.setattr(af2_backend_module.protein, "to_pdb", lambda prot: "PDB:TEMPLATE")

    af2_backend_module._write_processed_template_debug_artifacts(
        processed_feature_dict={
            "template_all_atom_positions": np.ones((2, 2, 37, 3), dtype=np.float32),
            "template_all_atom_mask": np.ones((2, 2, 37), dtype=np.float32),
            "template_aatype": np.stack([np.eye(22)[[0, 1]], np.eye(22)[[0, 1]]]),
            "template_domain_names": [b"good/template", b"bad"],
            "residue_index": np.array([7, 8], dtype=np.int32),
            "asym_id": np.array([3, 3], dtype=np.int32),
        },
        output_dir=tmp_path,
        model_name="modelB",
    )

    debug_dir = tmp_path / "templates_debug"
    assert (debug_dir / "modelB_good_template_idx0.pdb").is_file()
    error_file = debug_dir / "ERROR_modelB_template_1.txt"
    assert error_file.is_file()
    assert "boom" in error_file.read_text(encoding="utf-8")


def test_setup_configures_model_runners_and_validates_custom_names(af2_backend_module):
    configured = af2_backend_module.AlphaFold2Backend.setup(
        model_name="multimer",
        num_cycle=5,
        model_dir="/models",
        num_predictions_per_model=2,
        msa_depth_scan=True,
        model_names_custom=["model_1_multimer"],
        dropout=True,
    )

    runners = configured["model_runners"]
    assert sorted(runners) == [
        "model_1_multimer_pred_0_msa_16",
        "model_1_multimer_pred_1_msa_64",
    ]
    runner = runners["model_1_multimer_pred_0_msa_16"]
    assert runner.config["model"]["num_recycle"] == 5
    assert runner.config.model.global_config.eval_dropout is True
    assert runner.params["data_dir"] == "/models"

    with pytest.raises(Exception, match="Provided model names"):
        af2_backend_module.AlphaFold2Backend.setup(
            model_name="multimer",
            num_cycle=1,
            model_dir="/models",
            num_predictions_per_model=1,
            model_names_custom=["missing_model"],
        )


def test_predict_individual_job_writes_outputs_and_runs_debug_hooks(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AB"],
        feature_dict={"msa": np.ones((1, 2), dtype=np.int32)},
        multimeric_mode=True,
    )

    pad_calls = []
    msa_calls = []
    template_calls = []

    def fake_pad(feature_dict, desired_num_msa, desired_num_res):
        pad_calls.append((desired_num_res, desired_num_msa))

    processed_feature_dict = {
        "msa": np.ones((1, 2), dtype=np.int32),
        "template_all_atom_positions": np.ones((1, 2, 37, 3), dtype=np.float32),
        "template_all_atom_mask": np.ones((1, 2, 37), dtype=np.float32),
        "template_aatype": np.eye(22)[[0, 1]][None, :, :],
        "residue_index": np.array([0, 1], dtype=np.int32),
        "asym_id": np.array([1, 2], dtype=np.int32),
    }
    fake_runner = SimpleNamespace(
        multimer_mode=True,
        process_features=lambda feature_dict, random_seed: dict(processed_feature_dict),
        predict=lambda processed_feature_dict, random_seed: {
            "plddt": np.array([91.0, 88.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((2, 2), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
        },
    )

    monkeypatch.setattr(af2_backend_module, "pad_input_features", fake_pad)
    monkeypatch.setattr(
        af2_backend_module,
        "_write_processed_msa_debug_artifact",
        lambda **kwargs: msa_calls.append(kwargs["model_name"]),
    )
    monkeypatch.setattr(
        af2_backend_module,
        "_write_processed_template_debug_artifacts",
        lambda **kwargs: template_calls.append(kwargs["model_name"]),
    )

    results = af2_backend_module.AlphaFold2Backend.predict_individual_job(
        model_runners={"modelA": fake_runner},
        multimeric_object=multimer,
        allow_resume=False,
        skip_templates=False,
        output_dir=tmp_path,
        random_seed=7,
        desired_num_res=4,
        desired_num_msa=3,
        debug_msas=True,
        debug_templates=True,
    )

    assert pad_calls == [(4, 3)]
    assert msa_calls == ["modelA"]
    assert template_calls == ["modelA"]
    assert "modelA" in results
    assert results["modelA"]["seqs"] == ["AB"]
    assert (tmp_path / "result_modelA.pkl").is_file()
    assert (tmp_path / "unrelaxed_modelA.pdb").read_text(encoding="utf-8") == "PDB:predicted"
    timings = json.loads((tmp_path / "timings.json").read_text(encoding="utf-8"))
    assert "process_features_modelA" in timings
    assert "predict_and_compile_modelA" in timings
    with open(tmp_path / "result_modelA.pkl", "rb") as handle:
        payload = pickle.load(handle)
    assert payload["plddt"].tolist() == [91.0, 88.0]


def test_predict_individual_job_accepts_tuple_prediction_results(
    af2_backend_module,
    tmp_path,
):
    monomer = af2_backend_module.MonomericObject("single", "AB")
    monomer.feature_dict = {"residue_index": np.array([0, 1], dtype=np.int32)}

    fake_runner = SimpleNamespace(
        multimer_mode=False,
        process_features=lambda feature_dict, random_seed: dict(feature_dict),
        predict=lambda processed_feature_dict, random_seed: (
            {
                "plddt": np.array([91.0, 88.0], dtype=np.float32),
                "predicted_aligned_error": np.zeros((2, 2), dtype=np.float32),
                "max_predicted_aligned_error": 31.0,
            },
            {"auxiliary": "ignored"},
        ),
    )

    results = af2_backend_module.AlphaFold2Backend.predict_individual_job(
        model_runners={"modelA": fake_runner},
        multimeric_object=monomer,
        allow_resume=False,
        skip_templates=False,
        output_dir=tmp_path,
        random_seed=13,
    )

    assert results["modelA"]["plddt"].tolist() == [91.0, 88.0]
    assert results["modelA"]["seqs"] == ["AB"]
    assert results["modelA"]["unrelaxed_protein"].name == "predicted"
    with open(tmp_path / "result_modelA.pkl", "rb") as handle:
        payload = pickle.load(handle)
    assert payload["plddt"].tolist() == [91.0, 88.0]


def test_predict_individual_job_rejects_tuple_without_mapping_payload(
    af2_backend_module,
    tmp_path,
):
    monomer = af2_backend_module.MonomericObject("single", "AB")
    monomer.feature_dict = {"residue_index": np.array([0, 1], dtype=np.int32)}

    fake_runner = SimpleNamespace(
        multimer_mode=False,
        process_features=lambda feature_dict, random_seed: dict(feature_dict),
        predict=lambda processed_feature_dict, random_seed: (
            "not-a-mapping",
            {"auxiliary": "ignored"},
        ),
    )

    with pytest.raises(
        TypeError,
        match=r"model_runner\.predict must return a mapping or a \(mapping, auxiliary\) tuple",
    ):
        af2_backend_module.AlphaFold2Backend.predict_individual_job(
            model_runners={"modelA": fake_runner},
            multimeric_object=monomer,
            allow_resume=False,
            skip_templates=False,
            output_dir=tmp_path,
            random_seed=17,
        )


def test_predict_individual_job_rejects_skipped_templates_in_multimer_mode(
    af2_backend_module,
    tmp_path,
):
    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AB"],
        feature_dict={"msa": np.ones((1, 2), dtype=np.int32)},
        multimeric_mode=True,
    )
    fake_runner = SimpleNamespace(
        multimer_mode=True,
        process_features=lambda feature_dict, random_seed: {
            "seq_length": 2,
            "template_aatype": np.ones((1, 2), dtype=np.int32),
            "template_all_atom_positions": np.zeros((1, 2, 37, 3), dtype=np.float32),
            "template_all_atom_mask": np.ones((1, 2, 37), dtype=np.float32),
            "num_templates": np.array([1], dtype=np.int32),
        },
        predict=lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="cannot skip templates"):
        af2_backend_module.AlphaFold2Backend.predict_individual_job(
            model_runners={"modelA": fake_runner},
            multimeric_object=multimer,
            allow_resume=False,
            skip_templates=True,
            output_dir=tmp_path,
            random_seed=1,
        )


def test_predict_individual_job_resumes_completed_models_and_returns_early(
    af2_backend_module,
    tmp_path,
):
    monomer = af2_backend_module.MonomericObject("single", "AB")
    monomer.feature_dict = {"residue_index": np.array([0, 1], dtype=np.int32)}

    result_path = tmp_path / "result_modelA.pkl"
    with open(result_path, "wb") as handle:
        pickle.dump({"plddt": np.array([91.0, 88.0], dtype=np.float32)}, handle, protocol=4)
    (tmp_path / "unrelaxed_modelA.pdb").write_text("existing pdb", encoding="utf-8")

    process_calls = []
    predict_calls = []
    fake_runner = SimpleNamespace(
        multimer_mode=False,
        process_features=lambda feature_dict, random_seed: process_calls.append(
            random_seed
        )
        or dict(feature_dict),
        predict=lambda *args, **kwargs: predict_calls.append((args, kwargs)),
    )

    results = af2_backend_module.AlphaFold2Backend.predict_individual_job(
        model_runners={"modelA": fake_runner},
        multimeric_object=monomer,
        allow_resume=True,
        skip_templates=False,
        output_dir=tmp_path,
        random_seed=11,
    )

    assert predict_calls == []
    assert process_calls == [11]
    assert results["modelA"]["seqs"] == ["AB"]
    assert results["modelA"]["unrelaxed_protein"].name == "predicted"


def test_predict_individual_job_rejects_missing_or_zero_template_positions(
    af2_backend_module,
    tmp_path,
):
    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AB"],
        feature_dict={"msa": np.ones((1, 2), dtype=np.int32)},
        multimeric_mode=True,
    )
    missing_template_runner = SimpleNamespace(
        multimer_mode=True,
        process_features=lambda feature_dict, random_seed: {},
        predict=lambda *args, **kwargs: None,
    )
    zero_template_runner = SimpleNamespace(
        multimer_mode=True,
        process_features=lambda feature_dict, random_seed: {
            "template_all_atom_positions": np.zeros((1, 2, 37, 3), dtype=np.float32),
        },
        predict=lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="No template_all_atom_positions key found"):
        af2_backend_module.AlphaFold2Backend.predict_individual_job(
            model_runners={"modelA": missing_template_runner},
            multimeric_object=multimer,
            allow_resume=False,
            skip_templates=False,
            output_dir=tmp_path,
            random_seed=3,
        )

    with pytest.raises(ValueError, match="No valid templates found"):
        af2_backend_module.AlphaFold2Backend.predict_individual_job(
            model_runners={"modelA": zero_template_runner},
            multimeric_object=multimer,
            allow_resume=False,
            skip_templates=False,
            output_dir=tmp_path,
            random_seed=3,
        )


def test_predict_yields_results_for_each_object(af2_backend_module, monkeypatch, tmp_path):
    monomer_a = af2_backend_module.MonomericObject("a", "AA")
    monomer_b = af2_backend_module.MonomericObject("b", "BB")
    calls = []

    monkeypatch.setattr(
        af2_backend_module.AlphaFold2Backend,
        "predict_individual_job",
        staticmethod(
            lambda **kwargs: calls.append(kwargs["multimeric_object"].description)
            or {"modelA": kwargs["multimeric_object"].description}
        ),
    )

    outputs = list(
        af2_backend_module.AlphaFold2Backend.predict(
            model_runners={"modelA": object()},
            objects_to_model=[
                {"object": monomer_a, "output_dir": str(tmp_path / "a")},
                {"object": monomer_b, "output_dir": str(tmp_path / "b")},
            ],
            allow_resume=False,
            skip_templates=False,
            random_seed=5,
        )
    )

    assert calls == ["a", "b"]
    assert outputs == [
        {
            "object": monomer_a,
            "prediction_results": {"modelA": "a"},
            "output_dir": str(tmp_path / "a"),
        },
        {
            "object": monomer_b,
            "prediction_results": {"modelA": "b"},
            "output_dir": str(tmp_path / "b"),
        },
    ]


def test_recalculate_confidence_handles_multimer_and_monomer_paths(af2_backend_module):
    already_numpy = {
        "predicted_aligned_error": np.zeros((1, 1), dtype=np.float32),
        "plddt": np.array([42.0], dtype=np.float32),
    }
    assert (
        af2_backend_module.AlphaFold2Backend.recalculate_confidence(
            already_numpy, multimer_mode=False, total_num_res=1
        )
        is already_numpy
    )

    padded = {
        "predicted_aligned_error": {
            "logits": np.ones((4, 4), dtype=np.float32),
            "breaks": np.array([0.5], dtype=np.float32),
            "asym_id": np.array([0, 0, 1, 1], dtype=np.int32),
        },
        "plddt": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    }
    multimer_output = af2_backend_module.AlphaFold2Backend.recalculate_confidence(
        padded,
        multimer_mode=True,
        total_num_res=2,
    )
    assert multimer_output["ptm"] == 0.5
    assert multimer_output["iptm"] == 0.8
    assert multimer_output["ranking_confidence"] == pytest.approx(0.74)
    assert multimer_output["predicted_aligned_error"].shape == (2, 2)

    monomer_output = af2_backend_module.AlphaFold2Backend.recalculate_confidence(
        padded,
        multimer_mode=False,
        total_num_res=2,
    )
    assert monomer_output["ranking_confidence"] == pytest.approx(15.0)


def test_postprocess_ranks_models_relaxes_best_and_runs_cleanup(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    plot_calls = []
    cleanup_calls = []
    subprocess_calls = []

    monkeypatch.setattr(
        af2_backend_module,
        "plot_pae_from_matrix",
        lambda **kwargs: plot_calls.append(kwargs["ranking"]),
    )
    monkeypatch.setattr(
        af2_backend_module,
        "post_prediction_process",
        lambda *args, **kwargs: cleanup_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        af2_backend_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs))
        or SimpleNamespace(stderr="", stdout="", returncode=0),
    )

    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AA", "BB"],
        feature_dict={},
        multimeric_mode=True,
    )
    prediction_results = {
        "model_low": {
            "plddt": np.array([70.0, 71.0, 72.0, 73.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 0.2,
            "iptm": 0.1,
            "ptm": 0.2,
            "unrelaxed_protein": SimpleNamespace(name="low"),
            "seqs": ["AA", "BB"],
        },
        "model_high": {
            "plddt": np.array([90.0, 91.0, 92.0, 93.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 0.9,
            "iptm": 0.7,
            "ptm": 0.8,
            "unrelaxed_protein": SimpleNamespace(name="high"),
            "seqs": ["AA", "BB"],
        },
    }

    af2_backend_module.AlphaFold2Backend.postprocess(
        prediction_results=prediction_results,
        multimeric_object=multimer,
        output_dir=tmp_path,
        features_directory="/features",
        models_to_relax=af2_backend_module.ModelsToRelax.BEST,
        convert_to_modelcif=True,
    )

    ranking = json.loads((tmp_path / "ranking_debug.json").read_text(encoding="utf-8"))
    assert ranking["order"] == ["model_high", "model_low"]
    assert plot_calls == [0, 1]
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "RELAXED:high"
    assert (tmp_path / "ranked_1.pdb").read_text(encoding="utf-8") == "PDB:low"
    assert (tmp_path / "relax_metrics.json").is_file()
    assert cleanup_calls
    assert subprocess_calls


def test_postprocess_relaxes_all_using_saved_unrelaxed_pdbs(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    cleanup_calls = []
    monkeypatch.setattr(
        af2_backend_module,
        "plot_pae_from_matrix",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        af2_backend_module,
        "post_prediction_process",
        lambda *args, **kwargs: cleanup_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        af2_backend_module.protein,
        "from_pdb_string",
        lambda text: SimpleNamespace(name=text.strip()),
    )

    (tmp_path / "unrelaxed_model_a.pdb").write_text("model_a_unrelaxed", encoding="utf-8")
    (tmp_path / "unrelaxed_model_b.pdb").write_text("model_b_unrelaxed", encoding="utf-8")

    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AA", "BB"],
        feature_dict={},
        multimeric_mode=True,
    )
    prediction_results = {
        "model_a": {
            "plddt": np.array([80.0, 81.0, 82.0, 83.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 0.6,
            "iptm": 0.4,
            "ptm": 0.5,
            "seqs": ["AA", "BB"],
        },
        "model_b": {
            "plddt": np.array([90.0, 91.0, 92.0, 93.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 0.9,
            "iptm": 0.7,
            "ptm": 0.8,
            "seqs": ["AA", "BB"],
        },
    }

    af2_backend_module.AlphaFold2Backend.postprocess(
        prediction_results=prediction_results,
        multimeric_object=multimer,
        output_dir=tmp_path,
        features_directory="/features",
        models_to_relax=af2_backend_module.ModelsToRelax.ALL,
        convert_to_modelcif=False,
    )

    assert (tmp_path / "relaxed_model_a.pdb").read_text(encoding="utf-8") == "RELAXED:model_a_unrelaxed"
    assert (tmp_path / "relaxed_model_b.pdb").read_text(encoding="utf-8") == "RELAXED:model_b_unrelaxed"
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "RELAXED:model_b_unrelaxed"
    assert (tmp_path / "ranked_1.pdb").read_text(encoding="utf-8") == "RELAXED:model_a_unrelaxed"
    assert cleanup_calls


def test_postprocess_handles_monomers_without_relaxation_and_logs_modelcif_errors(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    cleanup_calls = []
    modelcif_errors = []
    plot_calls = []

    monkeypatch.setattr(
        af2_backend_module,
        "plot_pae_from_matrix",
        lambda **kwargs: plot_calls.append(kwargs["ranking"]),
    )
    monkeypatch.setattr(
        af2_backend_module,
        "post_prediction_process",
        lambda *args, **kwargs: cleanup_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        af2_backend_module.logging,
        "error",
        lambda message: modelcif_errors.append(message),
    )
    monkeypatch.setattr(
        af2_backend_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stderr="convert failed",
            stdout="",
            returncode=1,
        ),
    )

    monomer = af2_backend_module.MonomericObject("single", "AB")
    prediction_results = {
        "modelA": {
            "plddt": np.array([81.0, 82.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((2, 2), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 81.5,
            "ptm": 0.4,
            "seqs": ["AB"],
            "unrelaxed_protein": SimpleNamespace(name="mono"),
        }
    }

    af2_backend_module.AlphaFold2Backend.postprocess(
        prediction_results=prediction_results,
        multimeric_object=monomer,
        output_dir=tmp_path,
        features_directory="/features",
        models_to_relax=af2_backend_module.ModelsToRelax.NONE,
        convert_to_modelcif=True,
    )

    ranking = json.loads((tmp_path / "ranking_debug.json").read_text(encoding="utf-8"))
    assert ranking["plddts"] == {"modelA": 81.5}
    assert ranking["ptm"] == {"modelA": 0.4}
    assert ranking["order"] == ["modelA"]
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "PDB:mono"
    assert not (tmp_path / "relaxed_modelA.pdb").exists()
    assert plot_calls == [0]
    assert cleanup_calls
    assert modelcif_errors == ["Error: convert failed"]


def test_postprocess_skips_best_relaxation_below_score_threshold(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    info_messages = []
    monkeypatch.setattr(
        af2_backend_module,
        "plot_pae_from_matrix",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        af2_backend_module,
        "post_prediction_process",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        af2_backend_module.logging,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    multimer = af2_backend_module.MultimericObject(
        description="complex",
        input_seqs=["AA", "BB"],
        feature_dict={},
        multimeric_mode=True,
    )
    prediction_results = {
        "model_high": {
            "plddt": np.array([90.0, 91.0, 92.0, 93.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((4, 4), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 0.9,
            "iptm": 0.7,
            "ptm": 0.8,
            "unrelaxed_protein": SimpleNamespace(name="high"),
            "seqs": ["AA", "BB"],
        }
    }

    af2_backend_module.AlphaFold2Backend.postprocess(
        prediction_results=prediction_results,
        multimeric_object=multimer,
        output_dir=tmp_path,
        features_directory="/features",
        models_to_relax=af2_backend_module.ModelsToRelax.BEST,
        relax_best_score_threshold=0.8,
        convert_to_modelcif=False,
    )

    assert not (tmp_path / "relaxed_model_high.pdb").exists()
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "PDB:high"
    assert any("Skipping relaxation for model_high" in message for message in info_messages)


def test_postprocess_uses_ptm_threshold_for_best_monomer_relaxation(
    af2_backend_module,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        af2_backend_module,
        "plot_pae_from_matrix",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        af2_backend_module,
        "post_prediction_process",
        lambda *args, **kwargs: None,
    )

    monomer = af2_backend_module.MonomericObject("single", "AB")
    prediction_results = {
        "modelA": {
            "plddt": np.array([81.0, 82.0], dtype=np.float32),
            "predicted_aligned_error": np.zeros((2, 2), dtype=np.float32),
            "max_predicted_aligned_error": 31.0,
            "ranking_confidence": 81.5,
            "ptm": 0.4,
            "seqs": ["AB"],
            "unrelaxed_protein": SimpleNamespace(name="mono"),
        }
    }

    af2_backend_module.AlphaFold2Backend.postprocess(
        prediction_results=prediction_results,
        multimeric_object=monomer,
        output_dir=tmp_path,
        features_directory="/features",
        models_to_relax=af2_backend_module.ModelsToRelax.BEST,
        relax_best_score_threshold=0.3,
        convert_to_modelcif=False,
    )

    assert (tmp_path / "relaxed_modelA.pdb").read_text(encoding="utf-8") == "RELAXED:mono"
    assert (tmp_path / "ranked_0.pdb").read_text(encoding="utf-8") == "RELAXED:mono"
