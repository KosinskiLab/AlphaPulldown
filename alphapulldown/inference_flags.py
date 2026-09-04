"""One home for the inference flag set: which flags each backend accepts, and how the
parsed invocation becomes the configuration passed to ``backend.setup``.

Both facts used to be stated more than once. The per-backend flag sets were also copied
by hand into AlphaPulldownSnakemake's ``common.smk``, where they drifted. The
configuration dict was retyped verbatim in two places, and because every backend swallows
what it does not recognise in ``**kwargs``, a mistyped key in either copy was dropped in
silence rather than reported.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


# Accepted by every backend.
COMMON_FLAGS = frozenset({
    "input", "output_directory", "data_directory", "features_directory",
    "protein_delimiter", "fold_backend", "random_seed", "storage_mode",
})

# AlphaFold2 and the AlphaFold2-derived backends.
AF2_LIKE_FLAGS = frozenset({
    "compress_result_pickles", "remove_result_pickles", "models_to_relax",
    "relax_best_score_threshold", "remove_keys_from_pickles",
    "convert_to_modelcif", "allow_resume",
    "num_cycle", "num_predictions_per_model", "pair_msa",
    "save_features_for_multimeric_object", "skip_templates",
    "msa_depth_scan", "multimeric_template", "model_names", "msa_depth",
    "description_file", "path_to_mmt", "threshold_clashes", "hb_allowance",
    "plddt_threshold", "desired_num_res", "desired_num_msa",
    "benchmark", "model_preset", "use_ap_style", "use_gpu_relax", "dropout",
    # AF2 inference is JAX-compiled, so a persistent compile cache helps it too.
    "jax_compilation_cache_dir",
})

ALPHALINK_EXTRA_FLAGS = frozenset({"crosslinks"})

AF3_FLAGS = frozenset({
    "jax_compilation_cache_dir", "buckets", "flash_attention_implementation",
    "num_diffusion_samples", "num_seeds", "debug_templates", "debug_msas",
    "num_recycles", "save_embeddings", "save_distogram", "use_ap_style",
    "convert_to_modelcif",
})

FLAGS_BY_BACKEND: Mapping[str, frozenset] = {
    "alphafold2": COMMON_FLAGS | AF2_LIKE_FLAGS,
    "alphalink": COMMON_FLAGS | AF2_LIKE_FLAGS | ALPHALINK_EXTRA_FLAGS,
    "alphafold3": COMMON_FLAGS | AF3_FLAGS,
}


def unsupported_flags(backend_name: str, present: Iterable[str]) -> list[str]:
    """Flag names the named backend does not accept. Unknown backend: nothing to say."""
    allowed = FLAGS_BY_BACKEND.get(backend_name)
    if allowed is None:
        return []
    return sorted(set(present) - allowed)


# Configuration key -> attribute on the parsed invocation. One list, so a key can be
# mistyped in exactly one place and a wrong attribute name fails loudly at build time.
_MODEL_FLAG_SOURCES: Mapping[str, str] = {
    "num_cycle": "num_cycle",
    "model_dir": "data_directory",
    "num_predictions_per_model": "num_predictions_per_model",
    "crosslinks": "crosslinks",
    "desired_num_res": "desired_num_res",
    "desired_num_msa": "desired_num_msa",
    "skip_templates": "skip_templates",
    "allow_resume": "allow_resume",
    "num_diffusion_samples": "num_diffusion_samples",
    "num_recycles": "num_recycles",
    "return_embeddings": "save_embeddings",
    "return_distogram": "save_distogram",
    "flash_attention_implementation": "flash_attention_implementation",
    "buckets": "buckets",
    "jax_compilation_cache_dir": "jax_compilation_cache_dir",
    "features_directory": "features_directory",
    "num_seeds": "num_seeds",
    "debug_templates": "debug_templates",
    "debug_msas": "debug_msas",
    "dropout": "dropout",
}

_POSTPROCESS_FLAG_SOURCES: Mapping[str, str] = {
    "compress_pickles": "compress_result_pickles",
    "remove_pickles": "remove_result_pickles",
    "remove_keys_from_pickles": "remove_keys_from_pickles",
    "storage_mode": "storage_mode",
    "use_gpu_relax": "use_gpu_relax",
    "models_to_relax": "models_to_relax",
    "relax_best_score_threshold": "relax_best_score_threshold",
    "features_directory": "features_directory",
    "convert_to_modelcif": "convert_to_modelcif",
}

# Keys a backend's setup() may legitimately be given. Backends ignore what they do not
# use, so this set is the only thing standing between a mistyped key and silence.
MODEL_CONFIGURATION_KEYS = frozenset(_MODEL_FLAG_SOURCES) | {
    "model_name",
    # added for AlphaFold2 multimer batches
    "msa_depth_scan",
    "model_names_custom",
    "msa_depth",
}


def model_flags(flags: Any) -> Dict[str, Any]:
    """Configuration for ``backend.setup`` built from the parsed invocation."""
    configuration = {
        key: getattr(flags, attribute)
        for key, attribute in _MODEL_FLAG_SOURCES.items()
    }
    configuration["model_name"] = (
        "multimer_af2_crop" if flags.fold_backend == "alphalink" else "monomer_ptm"
    )
    return configuration


def postprocess_flags(flags: Any) -> Dict[str, Any]:
    """Configuration for ``backend.postprocess`` built from the parsed invocation."""
    return {
        key: getattr(flags, attribute)
        for key, attribute in _POSTPROCESS_FLAG_SOURCES.items()
    }


def validate_model_configuration(configuration: Mapping[str, Any]) -> None:
    """Reject configuration keys no backend consumes.

    ``setup(**configuration)`` lets every backend absorb keys meant for another one in
    ``**kwargs``, so without this a mistyped key is dropped without a word and the
    setting it was meant to carry simply never takes effect.
    """
    unknown = sorted(set(configuration) - MODEL_CONFIGURATION_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown model configuration key(s): {unknown}. "
            "Backends ignore keys they do not use, so this would otherwise be silent."
        )
