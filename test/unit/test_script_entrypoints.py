import importlib.util
import lzma
import os
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest


RUN_STRUCTURE_PREDICTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "scripts"
    / "run_structure_prediction.py"
)
RUN_MULTIMER_JOBS_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "scripts"
    / "run_multimer_jobs.py"
)


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


class _FakeFlag:
    def __init__(self, name, default):
        self.name = name
        self.value = default
        self.present = False
        self.using_default_value = True


class _FakeFlags:
    def __init__(self):
        object.__setattr__(self, "_flags", {})

    def define(self, name, default):
        flag = _FakeFlag(name, default)
        self._flags[name] = flag
        return flag

    def __call__(self, argv):
        return argv

    def __contains__(self, name):
        return name in self._flags

    def __getitem__(self, name):
        return self._flags[name]

    def __iter__(self):
        return iter(self._flags)

    def __getattr__(self, name):
        if name in self._flags:
            return self._flags[name].value
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._flags:
            flag = self._flags[name]
            flag.value = value
            flag.present = True
            flag.using_default_value = False
            return
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._flags:
            del self._flags[name]
            return
        object.__delattr__(self, name)

    def get_key_flags_for_module(self, _module):
        return list(self._flags.values())

    def flag_values_dict(self):
        return {name: flag.value for name, flag in self._flags.items()}


class _FakeFlagsModule(types.ModuleType):
    def __init__(self):
        super().__init__("absl.flags")
        self.FLAGS = _FakeFlags()

    def DEFINE_string(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def DEFINE_list(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def DEFINE_integer(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def DEFINE_float(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def DEFINE_boolean(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    DEFINE_bool = DEFINE_boolean

    def DEFINE_enum(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def DEFINE_enum_class(self, name, default, *_args, **_kwargs):
        return self.FLAGS.define(name, default)

    def mark_flag_as_required(self, *_args, **_kwargs):
        return None

    def mark_flags_as_required(self, *_args, **_kwargs):
        return None


def _restore_modules(saved_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _set_flag(flags_obj, name, value, *, present=True, using_default_value=False):
    flag = flags_obj[name]
    flag.value = value
    flag.present = present
    flag.using_default_value = using_default_value


def _load_run_structure_prediction_module():
    module_name = "test_run_structure_prediction_module"
    names_to_replace = [
        "absl",
        "absl.app",
        "absl.flags",
        "absl.logging",
        "jax",
        "alphapulldown",
        "alphapulldown.folding_backend",
        "alphapulldown.folding_backend.alphafold2_backend",
        "alphapulldown.objects",
        "alphapulldown.utils",
        "alphapulldown.utils.modelling_setup",
        "alphapulldown.utils.output_paths",
        module_name,
    ]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    flags_mod = _FakeFlagsModule()
    app_mod = types.ModuleType("absl.app")
    app_mod.run = lambda main: main
    logging_mod = types.ModuleType("absl.logging")
    logging_mod.INFO = 20
    logging_mod.info = lambda *args, **kwargs: None
    logging_mod.warning = lambda *args, **kwargs: None
    logging_mod.error = lambda *args, **kwargs: None
    logging_mod.set_verbosity = lambda *_args, **_kwargs: None

    absl_pkg = _package("absl")
    absl_pkg.app = app_mod
    absl_pkg.flags = flags_mod
    absl_pkg.logging = logging_mod

    jax_mod = types.ModuleType("jax")
    jax_mod.local_devices = lambda backend="gpu": []

    class ModelsToRelax(Enum):
        NONE = "none"
        ALL = "all"
        BEST = "best"

    root_pkg = _package("alphapulldown")
    folding_backend_mod = types.ModuleType("alphapulldown.folding_backend")
    folding_backend_mod.backend = SimpleNamespace()
    af2_backend_mod = types.ModuleType(
        "alphapulldown.folding_backend.alphafold2_backend"
    )
    af2_backend_mod.ModelsToRelax = ModelsToRelax

    objects_mod = types.ModuleType("alphapulldown.objects")

    class MonomericObject:
        def __init__(self, description, sequence):
            self.description = description
            self.sequence = sequence

    class ChoppedObject(MonomericObject):
        def __init__(self, description, sequence, monomeric_description=None):
            super().__init__(description, sequence)
            self.monomeric_description = monomeric_description or description

    class MultimericObject:
        feature_dict = {}

        def __init__(
            self,
            interactors,
            pair_msa,
            multimeric_template,
            multimeric_template_meta_data,
            multimeric_template_dir,
            threshold_clashes=1000,
            hb_allowance=0.4,
            plddt_threshold=0,
        ):
            self.interactors = list(interactors)
            self.pair_msa = pair_msa
            self.multimeric_template = multimeric_template
            self.multimeric_template_meta_data = multimeric_template_meta_data
            self.multimeric_template_dir = multimeric_template_dir
            self.threshold_clashes = threshold_clashes
            self.hb_allowance = hb_allowance
            self.plddt_threshold = plddt_threshold
            self.description = "_and_".join(interactor.description for interactor in interactors)
            self.input_seqs = [interactor.sequence for interactor in interactors]
            self.multimeric_mode = True

    objects_mod.MonomericObject = MonomericObject
    objects_mod.ChoppedObject = ChoppedObject
    objects_mod.MultimericObject = MultimericObject

    utils_pkg = _package("alphapulldown.utils")
    modelling_setup_mod = types.ModuleType("alphapulldown.utils.modelling_setup")
    modelling_setup_mod.create_interactors = lambda data, features_directory: []
    modelling_setup_mod.create_custom_info = lambda parsed: parsed
    modelling_setup_mod.parse_fold = lambda inputs, features_directory, delimiter: []

    output_paths_mod = types.ModuleType("alphapulldown.utils.output_paths")
    output_paths_mod.resolve_af3_combined_json_output_dir = (
        lambda json_inputs, out_dir, use_ap_style: out_dir
    )
    output_paths_mod.resolve_af3_json_output_dir = (
        lambda json_input, out_dir, use_ap_style, shared_output_root: out_dir
    )

    modules = {
        "absl": absl_pkg,
        "absl.app": app_mod,
        "absl.flags": flags_mod,
        "absl.logging": logging_mod,
        "jax": jax_mod,
        "alphapulldown": root_pkg,
        "alphapulldown.folding_backend": folding_backend_mod,
        "alphapulldown.folding_backend.alphafold2_backend": af2_backend_mod,
        "alphapulldown.objects": objects_mod,
        "alphapulldown.utils": utils_pkg,
        "alphapulldown.utils.modelling_setup": modelling_setup_mod,
        "alphapulldown.utils.output_paths": output_paths_mod,
    }
    for name, module in modules.items():
        sys.modules[name] = module

    root_pkg.folding_backend = folding_backend_mod
    root_pkg.objects = objects_mod
    root_pkg.utils = utils_pkg
    utils_pkg.modelling_setup = modelling_setup_mod
    utils_pkg.output_paths = output_paths_mod

    spec = importlib.util.spec_from_file_location(module_name, RUN_STRUCTURE_PREDICTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, saved_modules


def _load_run_multimer_jobs_module():
    module_name = "test_run_multimer_jobs_module"
    names_to_replace = [
        "absl",
        "absl.app",
        "absl.flags",
        "absl.logging",
        "jax",
        "alphapulldown",
        "alphapulldown.utils",
        "alphapulldown.utils.modelling_setup",
        "alphapulldown.utils.output_paths",
        "alphapulldown.scripts",
        "alphapulldown.scripts.run_structure_prediction",
        "alphapulldown_input_parser",
        module_name,
    ]
    saved_modules = {name: sys.modules.get(name) for name in names_to_replace}

    flags_mod = _FakeFlagsModule()
    app_mod = types.ModuleType("absl.app")
    app_mod.run = lambda main: main
    logging_mod = types.ModuleType("absl.logging")
    logging_mod.INFO = 20
    logging_mod.info = lambda *args, **kwargs: None
    logging_mod.warning = lambda *args, **kwargs: None
    logging_mod.error = lambda *args, **kwargs: None
    logging_mod.set_verbosity = lambda *_args, **_kwargs: None

    absl_pkg = _package("absl")
    absl_pkg.app = app_mod
    absl_pkg.flags = flags_mod
    absl_pkg.logging = logging_mod

    # Predefine the shared FLAGS that run_multimer_jobs expects from run_structure_prediction.
    shared_flag_defaults = {
        "models_to_relax": "NONE",
        "relax_best_score_threshold": None,
        "num_cycle": 3,
        "num_predictions_per_model": 1,
        "pair_msa": True,
        "msa_depth_scan": False,
        "multimeric_template": False,
        "model_names": None,
        "msa_depth": None,
        "crosslinks": None,
        "fold_backend": "alphafold2",
        "description_file": None,
        "path_to_mmt": None,
        "threshold_clashes": 1000,
        "hb_allowance": 0.4,
        "plddt_threshold": 0,
        "compress_result_pickles": False,
        "remove_result_pickles": False,
        "remove_keys_from_pickles": True,
        "use_ap_style": False,
        "use_gpu_relax": True,
        "protein_delimiter": "+",
        "desired_num_res": None,
        "desired_num_msa": None,
        "output_path": None,
        "data_dir": None,
        "monomer_objects_dir": None,
        "num_diffusion_samples": 5,
        "num_seeds": None,
        "flash_attention_implementation": "triton",
        "buckets": ["64", "128"],
        "jax_compilation_cache_dir": None,
        "save_embeddings": False,
        "save_distogram": False,
        "debug_templates": False,
        "debug_msas": False,
        "job_index": None,
    }
    for name, default in shared_flag_defaults.items():
        flags_mod.FLAGS.define(name, default)

    jax_mod = types.ModuleType("jax")
    jax_mod.local_devices = lambda backend="gpu": []

    root_pkg = _package("alphapulldown")
    utils_pkg = _package("alphapulldown.utils")
    scripts_pkg = _package("alphapulldown.scripts")
    run_structure_prediction_stub = types.ModuleType(
        "alphapulldown.scripts.run_structure_prediction"
    )
    run_structure_prediction_stub.FLAGS = flags_mod.FLAGS
    modelling_setup_mod = types.ModuleType("alphapulldown.utils.modelling_setup")
    modelling_setup_mod.parse_fold = (
        lambda input_list, features_directory, protein_delimiter: []
    )
    output_paths_mod = types.ModuleType("alphapulldown.utils.output_paths")
    output_paths_mod.derive_af3_job_name_from_json = (
        lambda json_input_path: Path(json_input_path).stem
    )

    input_parser_mod = types.ModuleType("alphapulldown_input_parser")
    input_parser_mod.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: []
    )

    modules = {
        "absl": absl_pkg,
        "absl.app": app_mod,
        "absl.flags": flags_mod,
        "absl.logging": logging_mod,
        "jax": jax_mod,
        "alphapulldown": root_pkg,
        "alphapulldown.utils": utils_pkg,
        "alphapulldown.utils.modelling_setup": modelling_setup_mod,
        "alphapulldown.utils.output_paths": output_paths_mod,
        "alphapulldown.scripts": scripts_pkg,
        "alphapulldown.scripts.run_structure_prediction": run_structure_prediction_stub,
        "alphapulldown_input_parser": input_parser_mod,
    }
    for name, module in modules.items():
        sys.modules[name] = module

    root_pkg.scripts = scripts_pkg
    root_pkg.utils = utils_pkg
    utils_pkg.modelling_setup = modelling_setup_mod
    utils_pkg.output_paths = output_paths_mod
    scripts_pkg.run_structure_prediction = run_structure_prediction_stub

    spec = importlib.util.spec_from_file_location(module_name, RUN_MULTIMER_JOBS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, saved_modules


@pytest.fixture
def run_structure_prediction_module():
    module, saved_modules = _load_run_structure_prediction_module()
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)
        _restore_modules(saved_modules)


@pytest.fixture
def run_multimer_jobs_module():
    module, saved_modules = _load_run_multimer_jobs_module()
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)
        _restore_modules(saved_modules)


def test_validate_flags_for_backend_rejects_disallowed_flags(run_structure_prediction_module):
    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold3")
    _set_flag(run_structure_prediction_module.FLAGS, "num_cycle", 3)

    with pytest.raises(ValueError, match="num_cycle"):
        run_structure_prediction_module._validate_flags_for_backend("alphafold3")


def test_validate_flags_for_backend_allows_unknown_backend(run_structure_prediction_module):
    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "custom-backend")
    _set_flag(run_structure_prediction_module.FLAGS, "num_cycle", 3)

    run_structure_prediction_module._validate_flags_for_backend("custom-backend")


def test_validate_flags_for_backend_falls_back_to_all_flags(
    run_structure_prediction_module,
    monkeypatch,
):
    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold3")
    _set_flag(run_structure_prediction_module.FLAGS, "num_cycle", 3)

    def _raise(_module):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        run_structure_prediction_module.FLAGS,
        "get_key_flags_for_module",
        _raise,
    )

    with pytest.raises(ValueError, match="num_cycle"):
        run_structure_prediction_module._validate_flags_for_backend("alphafold3")


def test_predict_structure_changes_backend_and_postprocesses_jobs(
    run_structure_prediction_module,
):
    calls = {"change_backend": [], "setup": [], "predict": [], "postprocess": []}

    class FakeBackend:
        def change_backend(self, **kwargs):
            calls["change_backend"].append(kwargs)

        def setup(self, **kwargs):
            calls["setup"].append(kwargs)
            return {"model_runners": {"modelA": object()}}

        def predict(self, **kwargs):
            calls["predict"].append(kwargs)
            return iter(
                [
                    {
                        "object": "obj1",
                        "prediction_results": {"modelA": "prediction"},
                        "output_dir": "/tmp/output",
                    }
                ]
            )

        def postprocess(self, **kwargs):
            calls["postprocess"].append(kwargs)

    run_structure_prediction_module.backend = FakeBackend()
    _set_flag(run_structure_prediction_module.FLAGS, "random_seed", 11)

    run_structure_prediction_module.predict_structure(
        objects_to_model=[{"object": "obj1", "output_dir": "/tmp/output"}],
        model_flags={"model_name": "monomer_ptm", "num_cycle": 3},
        postprocess_flags={"compress_pickles": True},
        fold_backend="alphafold2",
    )

    assert calls["change_backend"] == [{"backend_name": "alphafold2"}]
    assert calls["setup"] == [{"model_name": "monomer_ptm", "num_cycle": 3}]
    assert calls["predict"][0]["random_seed"] == 11
    assert calls["predict"][0]["objects_to_model"] == [
        {"object": "obj1", "output_dir": "/tmp/output"}
    ]
    assert calls["postprocess"] == [
        {
            "compress_pickles": True,
            "multimeric_object": "obj1",
            "prediction_results": {"modelA": "prediction"},
            "output_dir": "/tmp/output",
        }
    ]


@pytest.mark.parametrize(
    ("backend_name", "setup_payload", "expected_upper_bound"),
    [
        (
            "alphafold2",
            {"model_runners": {"modelA": object(), "modelB": object()}},
            sys.maxsize // 2,
        ),
        ("alphafold3", {"model_runners": {"modelA": object()}}, 2**32 - 1),
        ("custom-backend", {"model_runners": {"modelA": object()}}, sys.maxsize),
    ],
)
def test_predict_structure_generates_backend_specific_random_seeds(
    run_structure_prediction_module,
    monkeypatch,
    backend_name,
    setup_payload,
    expected_upper_bound,
):
    recorded_upper_bounds = []
    predict_calls = []

    class FakeBackend:
        def change_backend(self, **kwargs):
            return None

        def setup(self, **kwargs):
            return setup_payload

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            return iter([])

        def postprocess(self, **kwargs):
            raise AssertionError("postprocess should not run without predictions")

    run_structure_prediction_module.backend = FakeBackend()
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "random_seed",
        None,
        present=False,
        using_default_value=True,
    )
    monkeypatch.setattr(
        run_structure_prediction_module.random,
        "randrange",
        lambda upper: recorded_upper_bounds.append(upper) or 17,
    )

    run_structure_prediction_module.predict_structure(
        objects_to_model=[{"object": "obj1", "output_dir": "/tmp/output"}],
        model_flags={"model_name": "model"},
        postprocess_flags={},
        fold_backend=backend_name,
    )

    assert recorded_upper_bounds == [expected_upper_bound]
    assert predict_calls[0]["random_seed"] == 17


def test_pre_modelling_setup_decompresses_metadata_and_sets_input_sequences(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    metadata_path = feature_dir / "protA_feature_metadata_2026-03-30.json.xz"
    with lzma.open(metadata_path, "wt", encoding="utf-8") as handle:
        handle.write('{"meta": 1}')

    monomer = run_structure_prediction_module.MonomericObject("protA", "ACDE")
    returned_object, returned_output_dir = run_structure_prediction_module.pre_modelling_setup(
        [monomer],
        output_dir=str(tmp_path / "outputs"),
    )

    assert returned_object is monomer
    assert returned_object.input_seqs == ["ACDE"]
    copied_metadata = Path(returned_output_dir) / "protA_feature_metadata_2026-03-30.json"
    assert copied_metadata.read_text(encoding="utf-8") == '{"meta": 1}'


def test_pre_modelling_setup_saves_multimer_features_and_builds_unique_ap_style_dir(
    run_structure_prediction_module,
    monkeypatch,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", True)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", True)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (tmp_path / "outputs").mkdir()
    for description in ("protA", "protB"):
        (feature_dir / f"{description}_feature_metadata_2026-03-30.json").write_text(
            '{"meta": 1}',
            encoding="utf-8",
        )

    dumped = []
    monkeypatch.setattr(
        run_structure_prediction_module.pickle,
        "dump",
        lambda obj, handle: dumped.append((obj, handle.name)) or handle.close(),
    )

    monomer_a = run_structure_prediction_module.MonomericObject("protA", "AAAA")
    monomer_b = run_structure_prediction_module.MonomericObject("protB", "BBBB")
    returned_object, returned_output_dir = run_structure_prediction_module.pre_modelling_setup(
        [monomer_a, monomer_b],
        output_dir=str(tmp_path / "outputs"),
    )

    assert isinstance(returned_object, run_structure_prediction_module.MultimericObject)
    assert returned_output_dir.endswith("protA_and_protB")
    assert dumped == [
        (
            run_structure_prediction_module.MultimericObject.feature_dict,
            str(tmp_path / "outputs" / "multimeric_object_features.pkl"),
        )
    ]


def test_pre_modelling_setup_passes_multimeric_template_filters(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", True)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", "meta.csv")
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", "/tmp/templates")
    _set_flag(run_structure_prediction_module.FLAGS, "threshold_clashes", 12.5)
    _set_flag(run_structure_prediction_module.FLAGS, "hb_allowance", 0.7)
    _set_flag(run_structure_prediction_module.FLAGS, "plddt_threshold", 42.0)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for description in ("protA", "protB"):
        (feature_dir / f"{description}_feature_metadata_2026-03-30.json").write_text(
            '{"meta": 1}',
            encoding="utf-8",
        )

    monomer_a = run_structure_prediction_module.MonomericObject("protA", "AAAA")
    monomer_b = run_structure_prediction_module.MonomericObject("protB", "BBBB")
    returned_object, _ = run_structure_prediction_module.pre_modelling_setup(
        [monomer_a, monomer_b],
        output_dir=str(tmp_path / "outputs"),
    )

    assert returned_object.threshold_clashes == 12.5
    assert returned_object.hb_allowance == 0.7
    assert returned_object.plddt_threshold == 42.0


def test_pre_modelling_setup_builds_ap_style_homo_oligomer_dir(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", True)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "protA_feature_metadata_2026-03-30.json").write_text(
        '{"meta": 1}',
        encoding="utf-8",
    )

    monomer_a = run_structure_prediction_module.MonomericObject("protA", "AAAA")
    monomer_b = run_structure_prediction_module.MonomericObject("protA", "AAAA")
    returned_object, returned_output_dir = run_structure_prediction_module.pre_modelling_setup(
        [monomer_a, monomer_b],
        output_dir=str(tmp_path / "outputs"),
    )

    assert isinstance(returned_object, run_structure_prediction_module.MultimericObject)
    assert returned_output_dir.endswith("protA_homo_2er")
    assert Path(returned_output_dir).is_dir()


def test_pre_modelling_setup_warns_for_long_paths_and_uses_chopped_metadata_name(
    run_structure_prediction_module,
    monkeypatch,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        ["/features"],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    warnings = []
    glob_patterns = []
    created_dirs = []
    monkeypatch.setattr(
        run_structure_prediction_module.glob,
        "glob",
        lambda pattern: glob_patterns.append(pattern) or [],
    )
    monkeypatch.setattr(
        run_structure_prediction_module.logging,
        "warning",
        lambda message: warnings.append(message),
    )
    monkeypatch.setattr(
        run_structure_prediction_module.os,
        "makedirs",
        lambda path, exist_ok=True: created_dirs.append(path),
    )

    chopped = run_structure_prediction_module.ChoppedObject(
        "fragmentA",
        "ACDE",
        monomeric_description="protA",
    )
    long_output_dir = "a" * 4100
    returned_object, returned_output_dir = run_structure_prediction_module.pre_modelling_setup(
        [chopped],
        output_dir=long_output_dir,
    )

    assert returned_object is chopped
    assert returned_output_dir == long_output_dir
    assert glob_patterns == ["/features/protA_feature_metadata_*.json*"]
    assert created_dirs == [long_output_dir]
    assert any("Output directory path is too long" in message for message in warnings)
    assert any("No feature metadata found for fragmentA" in message for message in warnings)


def test_pre_modelling_setup_allows_skip_msa_monomers_with_default_pair_flag(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "protA_feature_metadata_2026-03-30.json").write_text(
        '{"meta": 1}',
        encoding="utf-8",
    )

    monomer = run_structure_prediction_module.MonomericObject("protA", "ACDE")
    monomer.skip_msa = True

    returned_object, _ = run_structure_prediction_module.pre_modelling_setup(
        [monomer],
        output_dir=str(tmp_path / "outputs"),
    )

    assert returned_object is monomer
    assert returned_object.input_seqs == ["ACDE"]


def test_pre_modelling_setup_rejects_pair_msa_for_skip_msa_multimers(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", True)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for description in ("protA", "protB"):
        (feature_dir / f"{description}_feature_metadata_2026-03-30.json").write_text(
            '{"meta": 1}',
            encoding="utf-8",
        )

    monomer_a = run_structure_prediction_module.MonomericObject("protA", "ACDE")
    monomer_a.skip_msa = True
    monomer_b = run_structure_prediction_module.MonomericObject("protB", "BCDE")

    with pytest.raises(ValueError, match="--pair_msa=False"):
        run_structure_prediction_module.pre_modelling_setup(
            [monomer_a, monomer_b],
            output_dir=str(tmp_path / "outputs"),
        )


def test_pre_modelling_setup_allows_skip_msa_when_pairing_disabled(
    run_structure_prediction_module,
    tmp_path,
):
    _set_flag(run_structure_prediction_module.FLAGS, "pair_msa", False)
    _set_flag(run_structure_prediction_module.FLAGS, "multimeric_template", False)
    _set_flag(run_structure_prediction_module.FLAGS, "description_file", None)
    _set_flag(run_structure_prediction_module.FLAGS, "path_to_mmt", None)
    _set_flag(run_structure_prediction_module.FLAGS, "save_features_for_multimeric_object", False)
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for description in ("protA", "protB"):
        (feature_dir / f"{description}_feature_metadata_2026-03-30.json").write_text(
            '{"meta": 1}',
            encoding="utf-8",
        )

    monomer_a = run_structure_prediction_module.MonomericObject("protA", "AAAA")
    monomer_a.skip_msa = True
    monomer_b = run_structure_prediction_module.MonomericObject("protB", "BBBB")

    returned_object, _ = run_structure_prediction_module.pre_modelling_setup(
        [monomer_a, monomer_b],
        output_dir=str(tmp_path / "outputs"),
    )

    assert isinstance(returned_object, run_structure_prediction_module.MultimericObject)
    assert returned_object.pair_msa is False


def test_main_routes_protein_and_json_jobs_to_predict_structure(
    run_structure_prediction_module,
    monkeypatch,
    tmp_path,
):
    captured_calls = []
    protein_obj = run_structure_prediction_module.MonomericObject("protA", "AC")
    multimer_obj = run_structure_prediction_module.MultimericObject(
        [protein_obj, protein_obj],
        pair_msa=True,
        multimeric_template=False,
        multimeric_template_meta_data=None,
        multimeric_template_dir=None,
    )

    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold3")
    _set_flag(run_structure_prediction_module.FLAGS, "input", ["job1", "job2"])
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "output_directory",
        [str(tmp_path / "shared-output")],
    )
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "protein_delimiter", "+")
    _set_flag(run_structure_prediction_module.FLAGS, "data_directory", "/models")
    _set_flag(run_structure_prediction_module.FLAGS, "num_diffusion_samples", 5)
    _set_flag(run_structure_prediction_module.FLAGS, "num_recycles", 10)
    _set_flag(run_structure_prediction_module.FLAGS, "save_embeddings", False)
    _set_flag(run_structure_prediction_module.FLAGS, "save_distogram", False)
    _set_flag(run_structure_prediction_module.FLAGS, "flash_attention_implementation", "triton")
    _set_flag(run_structure_prediction_module.FLAGS, "buckets", ["64", "128"])
    _set_flag(run_structure_prediction_module.FLAGS, "jax_compilation_cache_dir", None)
    _set_flag(run_structure_prediction_module.FLAGS, "num_seeds", None)
    _set_flag(run_structure_prediction_module.FLAGS, "debug_templates", False)
    _set_flag(run_structure_prediction_module.FLAGS, "debug_msas", False)
    _set_flag(run_structure_prediction_module.FLAGS, "use_ap_style", True)

    monkeypatch.setattr(
        run_structure_prediction_module,
        "parse_fold",
        lambda inputs, features_directory, delimiter: [["parsed"]],
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "create_custom_info",
        lambda parsed: "data",
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "create_interactors",
        lambda data, features_directory: [
            [protein_obj],
            [{"json_input": "/tmp/job.json"}],
        ],
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "pre_modelling_setup",
        lambda prot_objs, output_dir: (multimer_obj, f"{output_dir}/protein"),
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "resolve_af3_json_output_dir",
        lambda json_input, out_dir, use_ap_style, shared_output_root: f"{out_dir}/json",
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "predict_structure",
        lambda **kwargs: captured_calls.append(kwargs),
    )

    run_structure_prediction_module.main([])

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["fold_backend"] == "alphafold3"
    assert call["objects_to_model"] == [
        {"object": multimer_obj, "output_dir": f"{tmp_path / 'shared-output'}/protein"},
        {"object": {"json_input": "/tmp/job.json"}, "output_dir": f"{tmp_path / 'shared-output'}/json"},
    ]
    assert call["model_flags"]["model_name"] == "multimer"


def test_main_sets_multimer_model_flags_for_multimer_jobs(
    run_structure_prediction_module,
    monkeypatch,
    tmp_path,
):
    captured_calls = []
    protein_obj = run_structure_prediction_module.MonomericObject("protA", "AC")
    multimer_obj = run_structure_prediction_module.MultimericObject(
        [protein_obj, protein_obj],
        pair_msa=True,
        multimeric_template=False,
        multimeric_template_meta_data=None,
        multimeric_template_dir=None,
    )

    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_structure_prediction_module.FLAGS, "input", ["job1"])
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "output_directory",
        [str(tmp_path / "shared-output")],
    )
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        [str(tmp_path / "features")],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "protein_delimiter", "+")
    _set_flag(run_structure_prediction_module.FLAGS, "data_directory", "/models")
    _set_flag(run_structure_prediction_module.FLAGS, "msa_depth_scan", True)
    _set_flag(run_structure_prediction_module.FLAGS, "model_names", ["model_2_multimer_v3"])
    _set_flag(run_structure_prediction_module.FLAGS, "msa_depth", 64)
    _set_flag(run_structure_prediction_module.FLAGS, "relax_best_score_threshold", 0.6)

    monkeypatch.setattr(run_structure_prediction_module, "parse_fold", lambda *args: [["parsed"]])
    monkeypatch.setattr(run_structure_prediction_module, "create_custom_info", lambda parsed: "data")
    monkeypatch.setattr(
        run_structure_prediction_module,
        "create_interactors",
        lambda data, features_directory: [[protein_obj, protein_obj]],
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "pre_modelling_setup",
        lambda prot_objs, output_dir: (multimer_obj, str(tmp_path / "protein")),
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "predict_structure",
        lambda **kwargs: captured_calls.append(kwargs),
    )

    run_structure_prediction_module.main([])

    assert len(captured_calls) == 1
    assert captured_calls[0]["model_flags"]["model_name"] == "multimer"
    assert captured_calls[0]["model_flags"]["msa_depth_scan"] is True
    assert captured_calls[0]["model_flags"]["model_names_custom"] == ["model_2_multimer_v3"]
    assert captured_calls[0]["model_flags"]["msa_depth"] == 64
    assert captured_calls[0]["postprocess_flags"]["relax_best_score_threshold"] == 0.6


def test_main_rejects_mismatched_output_directories(
    run_structure_prediction_module,
    monkeypatch,
):
    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_structure_prediction_module.FLAGS, "input", ["job1", "job2"])
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "output_directory",
        ["/tmp/out1", "/tmp/out2", "/tmp/out3"],
    )
    _set_flag(
        run_structure_prediction_module.FLAGS,
        "features_directory",
        ["/tmp/features"],
    )
    _set_flag(run_structure_prediction_module.FLAGS, "protein_delimiter", "+")

    monkeypatch.setattr(run_structure_prediction_module, "parse_fold", lambda *args: [])
    monkeypatch.setattr(run_structure_prediction_module, "create_custom_info", lambda parsed: parsed)
    monkeypatch.setattr(run_structure_prediction_module, "create_interactors", lambda data, features: [])

    with pytest.raises(ValueError, match="Either specify one output_directory"):
        run_structure_prediction_module.main([])


def test_main_skips_empty_interactor_groups_without_predicting(
    run_structure_prediction_module,
    monkeypatch,
):
    predict_calls = []

    _set_flag(run_structure_prediction_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_structure_prediction_module.FLAGS, "input", ["job1"])
    _set_flag(run_structure_prediction_module.FLAGS, "output_directory", ["/tmp/out"])
    _set_flag(run_structure_prediction_module.FLAGS, "features_directory", ["/tmp/features"])
    _set_flag(run_structure_prediction_module.FLAGS, "protein_delimiter", "+")
    _set_flag(run_structure_prediction_module.FLAGS, "data_directory", "/models")

    monkeypatch.setattr(run_structure_prediction_module, "parse_fold", lambda *args: [["parsed"]])
    monkeypatch.setattr(run_structure_prediction_module, "create_custom_info", lambda parsed: parsed)
    monkeypatch.setattr(
        run_structure_prediction_module,
        "create_interactors",
        lambda data, features_directory: [[]],
    )
    monkeypatch.setattr(
        run_structure_prediction_module,
        "predict_structure",
        lambda **kwargs: predict_calls.append(kwargs),
    )

    run_structure_prediction_module.main([])

    assert predict_calls == []


def test_run_multimer_jobs_dry_run_exits_and_reports_count(run_multimer_jobs_module):
    messages = []
    run_multimer_jobs_module.logging.info = messages.append
    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", True)
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: ["A,B", "C,D"]
    )

    with pytest.raises(SystemExit) as exc:
        run_multimer_jobs_module.main(["prog"])

    assert exc.value.code == 0
    assert messages == ["Dry run: the total number of jobs to be run: 2"]


def test_run_multimer_jobs_builds_af3_commands_and_sanitizes_env(
    run_multimer_jobs_module,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        run_multimer_jobs_module.subprocess,
        "run",
        lambda command, check, env: calls.append((command, env)),
    )
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: ["A,B", "C;D"]
    )

    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", False)
    _set_flag(run_multimer_jobs_module.FLAGS, "fold_backend", "alphafold3")
    _set_flag(run_multimer_jobs_module.FLAGS, "output_path", "/tmp/output")
    _set_flag(run_multimer_jobs_module.FLAGS, "data_dir", "/tmp/models")
    _set_flag(run_multimer_jobs_module.FLAGS, "monomer_objects_dir", ["/tmp/features"])
    _set_flag(run_multimer_jobs_module.FLAGS, "num_cycle", 7)
    _set_flag(run_multimer_jobs_module.FLAGS, "num_diffusion_samples", 9)
    _set_flag(run_multimer_jobs_module.FLAGS, "num_seeds", 3)
    _set_flag(run_multimer_jobs_module.FLAGS, "save_embeddings", True)
    _set_flag(run_multimer_jobs_module.FLAGS, "save_distogram", True)
    _set_flag(run_multimer_jobs_module.FLAGS, "debug_templates", True)
    _set_flag(run_multimer_jobs_module.FLAGS, "debug_msas", True)
    run_multimer_jobs_module.FLAGS["use_ap_style"].value = False
    run_multimer_jobs_module.FLAGS["use_ap_style"].present = False
    run_multimer_jobs_module.FLAGS["use_ap_style"].using_default_value = True

    original_xla_client = os.environ.get("XLA_CLIENT_MEM_FRACTION")
    original_xla_python = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
    try:
        run_multimer_jobs_module.main(["prog"])
    finally:
        if original_xla_client is None:
            os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
        else:
            os.environ["XLA_CLIENT_MEM_FRACTION"] = original_xla_client
        if original_xla_python is None:
            os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        else:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = original_xla_python

    assert len(calls) == 2
    first_command, first_env = calls[0]
    assert "--num_recycles" in first_command
    assert "--num_cycle" not in first_command
    assert "--use_ap_style" in first_command
    assert "--save_embeddings" in first_command
    assert "--save_distogram" in first_command
    assert "--debug_templates" in first_command
    assert "--debug_msas" in first_command
    assert "--input" in first_command
    assert "A:B" in first_command
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in first_env


def test_run_multimer_jobs_scopes_af3_json_jobs_to_per_job_dirs(
    run_multimer_jobs_module,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        run_multimer_jobs_module.subprocess,
        "run",
        lambda command, check, env: calls.append((command, env)),
    )
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: [
            "protein_with_ptms.json",
            "P01308_af3_input.json",
        ]
    )
    run_multimer_jobs_module.parse_fold = (
        lambda input_list, features_directory, protein_delimiter: [
            [{"json_input": f"/tmp/features/{input_list[0]}"}]
        ]
    )
    run_multimer_jobs_module.derive_af3_job_name_from_json = (
        lambda json_input_path: {
            "/tmp/features/protein_with_ptms.json": "protein_with_ptms",
            "/tmp/features/P01308_af3_input.json": "p01308",
        }[json_input_path]
    )

    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", False)
    _set_flag(run_multimer_jobs_module.FLAGS, "fold_backend", "alphafold3")
    _set_flag(run_multimer_jobs_module.FLAGS, "output_path", "/tmp/output")
    _set_flag(run_multimer_jobs_module.FLAGS, "data_dir", "/tmp/models")
    _set_flag(run_multimer_jobs_module.FLAGS, "monomer_objects_dir", ["/tmp/features"])
    _set_flag(run_multimer_jobs_module.FLAGS, "use_ap_style", True)

    run_multimer_jobs_module.main(["prog"])

    assert len(calls) == 2
    first_command, _ = calls[0]
    second_command, _ = calls[1]
    assert first_command[first_command.index("--output_directory") + 1] == (
        "/tmp/output/protein_with_ptms"
    )
    assert second_command[second_command.index("--output_directory") + 1] == (
        "/tmp/output/p01308"
    )


def test_run_multimer_jobs_combines_inputs_when_padding_requested(
    run_multimer_jobs_module,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        run_multimer_jobs_module.subprocess,
        "run",
        lambda command, check, env: calls.append(command),
    )
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: ["job1", "job2"]
    )

    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", False)
    _set_flag(run_multimer_jobs_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_multimer_jobs_module.FLAGS, "output_path", "/tmp/output")
    _set_flag(run_multimer_jobs_module.FLAGS, "data_dir", "/tmp/models")
    _set_flag(run_multimer_jobs_module.FLAGS, "monomer_objects_dir", ["/tmp/features"])
    _set_flag(run_multimer_jobs_module.FLAGS, "desired_num_res", 256)
    _set_flag(run_multimer_jobs_module.FLAGS, "desired_num_msa", 128)
    _set_flag(run_multimer_jobs_module.FLAGS, "pair_msa", False)

    run_multimer_jobs_module.main(["prog"])

    assert len(calls) == 1
    assert "--input" in calls[0]
    input_index = calls[0].index("--input")
    assert calls[0][input_index + 1] == "job1,job2"
    assert "--nopair_msa" in calls[0]


def test_run_multimer_jobs_forwards_multimeric_template_filters(
    run_multimer_jobs_module,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        run_multimer_jobs_module.subprocess,
        "run",
        lambda command, check, env: calls.append(command),
    )
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: ["job1"]
    )

    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", False)
    _set_flag(run_multimer_jobs_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_multimer_jobs_module.FLAGS, "output_path", "/tmp/output")
    _set_flag(run_multimer_jobs_module.FLAGS, "data_dir", "/tmp/models")
    _set_flag(run_multimer_jobs_module.FLAGS, "monomer_objects_dir", ["/tmp/features"])
    _set_flag(run_multimer_jobs_module.FLAGS, "multimeric_template", True)
    _set_flag(run_multimer_jobs_module.FLAGS, "threshold_clashes", 12.5)
    _set_flag(run_multimer_jobs_module.FLAGS, "hb_allowance", 0.7)
    _set_flag(run_multimer_jobs_module.FLAGS, "plddt_threshold", 42.0)

    run_multimer_jobs_module.main(["prog"])

    assert len(calls) == 1
    assert "--threshold_clashes" in calls[0]
    assert calls[0][calls[0].index("--threshold_clashes") + 1] == "12.5"
    assert "--hb_allowance" in calls[0]
    assert calls[0][calls[0].index("--hb_allowance") + 1] == "0.7"
    assert "--plddt_threshold" in calls[0]
    assert calls[0][calls[0].index("--plddt_threshold") + 1] == "42.0"


def test_run_multimer_jobs_forwards_relax_best_score_threshold(
    run_multimer_jobs_module,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        run_multimer_jobs_module.subprocess,
        "run",
        lambda command, check, env: calls.append(command),
    )
    run_multimer_jobs_module.generate_fold_specifications = (
        lambda input_files, delimiter, exclude_permutations: ["job1"]
    )

    _set_flag(run_multimer_jobs_module.FLAGS, "mode", "custom")
    _set_flag(run_multimer_jobs_module.FLAGS, "protein_lists", ["proteins.txt"])
    _set_flag(run_multimer_jobs_module.FLAGS, "dry_run", False)
    _set_flag(run_multimer_jobs_module.FLAGS, "fold_backend", "alphafold2")
    _set_flag(run_multimer_jobs_module.FLAGS, "output_path", "/tmp/output")
    _set_flag(run_multimer_jobs_module.FLAGS, "data_dir", "/tmp/models")
    _set_flag(run_multimer_jobs_module.FLAGS, "monomer_objects_dir", ["/tmp/features"])
    _set_flag(run_multimer_jobs_module.FLAGS, "models_to_relax", "Best")
    _set_flag(run_multimer_jobs_module.FLAGS, "relax_best_score_threshold", 0.6)

    run_multimer_jobs_module.main(["prog"])

    assert len(calls) == 1
    assert "--relax_best_score_threshold" in calls[0]
    assert calls[0][calls[0].index("--relax_best_score_threshold") + 1] == "0.6"
