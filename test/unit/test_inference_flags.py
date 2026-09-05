"""The inference flag set has one home; these pin what that home guarantees."""

import pytest

from alphapulldown import inference_flags


class _Flags:
    """Stand-in for the parsed invocation."""

    def __init__(self, **overrides):
        defaults = dict(
            num_cycle=3, data_directory="/weights", num_predictions_per_model=1,
            crosslinks=None, desired_num_res=None, desired_num_msa=None,
            skip_templates=False, allow_resume=True, num_diffusion_samples=5,
            num_recycles=10, save_embeddings=False, save_distogram=False,
            flash_attention_implementation="triton", buckets=["256"],
            jax_compilation_cache_dir=None, features_directory=["/features"],
            num_seeds=None, debug_templates=False, debug_msas=False, dropout=False,
            fold_backend="alphafold3",
            compress_result_pickles=False, remove_result_pickles=False,
            remove_keys_from_pickles=None, storage_mode="default",
            use_gpu_relax=True, models_to_relax="none",
            relax_best_score_threshold=None, convert_to_modelcif=True,
        )
        defaults.update(overrides)
        self.__dict__.update(defaults)


def test_every_backend_accepts_the_common_flags():
    for backend, allowed in inference_flags.FLAGS_BY_BACKEND.items():
        assert inference_flags.COMMON_FLAGS <= allowed, backend


def test_af2_only_and_af3_only_flags_are_rejected_by_the_other_backend():
    assert inference_flags.unsupported_flags("alphafold3", ["allow_resume"]) == [
        "allow_resume"
    ]
    assert inference_flags.unsupported_flags("alphafold2", ["buckets"]) == ["buckets"]


def test_jax_compile_cache_is_accepted_by_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        assert inference_flags.unsupported_flags(
            backend, ["jax_compilation_cache_dir"]
        ) == []


def test_convert_to_modelcif_is_accepted_by_both_backends():
    # The workflow's hand-copied table omitted this and produced a false warning.
    for backend in ("alphafold2", "alphafold3"):
        assert inference_flags.unsupported_flags(backend, ["convert_to_modelcif"]) == []


def test_unknown_backend_reports_nothing():
    assert inference_flags.unsupported_flags("boltz", ["anything"]) == []


def test_model_flags_names_the_alphalink_model():
    assert inference_flags.model_flags(_Flags())["model_name"] == "monomer_ptm"
    alphalink = inference_flags.model_flags(_Flags(fold_backend="alphalink"))
    assert alphalink["model_name"] == "multimer_af2_crop"


def test_built_configuration_is_accepted_by_its_own_validator():
    inference_flags.validate_model_configuration(inference_flags.model_flags(_Flags()))


def test_multimer_extras_are_accepted():
    configuration = inference_flags.model_flags(_Flags())
    configuration.update(
        model_name="multimer", msa_depth_scan=False,
        model_names_custom=None, msa_depth=None,
    )
    inference_flags.validate_model_configuration(configuration)


def test_mistyped_configuration_key_is_reported_not_dropped():
    configuration = inference_flags.model_flags(_Flags())
    configuration["jax_compilation_cache_dirr"] = "/cache"
    with pytest.raises(ValueError, match="jax_compilation_cache_dirr"):
        inference_flags.validate_model_configuration(configuration)


def test_postprocess_flags_are_built_from_the_invocation():
    built = inference_flags.postprocess_flags(_Flags(use_gpu_relax=False))
    assert built["use_gpu_relax"] is False
    assert built["compress_pickles"] is False
