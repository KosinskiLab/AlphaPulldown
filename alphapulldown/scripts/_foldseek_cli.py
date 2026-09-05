"""Shared flag schema for the Foldseek + ESMFold structural template path.

One place defines these flags so the feature entry point and the standalone
structure-prediction stage cannot drift apart, in the same way
``_mmseqs2_cli`` holds the local MMseqs2 flags.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from absl import flags

from alphapulldown.structural_templates import (
    ALIGNMENT_TYPE_3DI_AA,
    EsmfoldSettings,
    EsmfoldStructurePredictor,
    FoldseekSearchSettings,
    FoldseekTemplateSearcher,
    PredictedStructureCache,
    StructureDatabaseSpec,
    SubprocessFoldseekProcess,
)


# Every flag this module defines. Recorded once so the metadata filter and the
# validator cannot fall behind the definitions.
STRUCTURAL_TEMPLATE_FLAG_NAMES = (
    "use_foldseek_templates",
    "foldseek_binary_path",
    "foldseek_database_path",
    "foldseek_database_id",
    "foldseek_temp_dir",
    "foldseek_e_value",
    "foldseek_max_hits",
    "foldseek_min_alignment_tm_score",
    "foldseek_alignment_type",
    "foldseek_threads",
    "structural_template_cache_dir",
    "esmfold_model_dir",
    "esmfold_device",
    "esmfold_chunk_size",
    "esmfold_max_sequence_length",
)

# Without these the search cannot be configured at all.
REQUIRED_WHEN_ENABLED = (
    "foldseek_database_path",
    "foldseek_database_id",
    "esmfold_model_dir",
)


def _define_once(name: str, define: Callable, default, help_text: str) -> None:
    if name not in flags.FLAGS:
        define(name, default, help_text)


def define_structural_template_flags(*, include_switch: bool = True) -> None:
    """Register the structural-template flags without duplicating global names."""
    if include_switch:
        _define_once(
            "use_foldseek_templates",
            flags.DEFINE_boolean,
            False,
            "Find templates by structure instead of by sequence: predict the query "
            "fold with ESMFold and search it against a local Foldseek database. "
            "Off by default; the sequence-based search is unchanged when it is off.",
        )
    _define_once(
        "foldseek_binary_path",
        flags.DEFINE_string,
        shutil.which("foldseek"),
        "Path to the local Foldseek executable.",
    )
    _define_once(
        "foldseek_database_path",
        flags.DEFINE_string,
        None,
        "Prefix of the local Foldseek structure database to search. Its chains "
        "must exist in --template_mmcif_dir, so building it from that directory "
        "is the safe choice.",
    )
    _define_once(
        "foldseek_database_id",
        flags.DEFINE_string,
        None,
        "Immutable identifier for the Foldseek database build. Cached alignments "
        "are reused only when it matches.",
    )
    _define_once(
        "foldseek_temp_dir",
        flags.DEFINE_string,
        None,
        "Fast local scratch directory for Foldseek. Defaults to a 'tmp' "
        "subdirectory of --structural_template_cache_dir.",
    )
    _define_once(
        "foldseek_e_value",
        flags.DEFINE_float,
        1e-3,
        "Foldseek search E-value cutoff.",
    )
    _define_once(
        "foldseek_max_hits",
        flags.DEFINE_integer,
        100,
        "Maximum structures Foldseek may return per query. The featuriser keeps "
        "far fewer, but it discards hits that fail its prefilters.",
    )
    _define_once(
        "foldseek_min_alignment_tm_score",
        flags.DEFINE_float,
        0.0,
        "Discard hits whose alignment TM-score is below this value (0 keeps all).",
    )
    _define_once(
        "foldseek_alignment_type",
        flags.DEFINE_integer,
        ALIGNMENT_TYPE_3DI_AA,
        "Foldseek alignment mode: 1 for TMalign, 2 for 3Di+AA.",
    )
    _define_once(
        "foldseek_threads",
        flags.DEFINE_integer,
        8,
        "CPU threads for Foldseek.",
    )
    _define_once(
        "structural_template_cache_dir",
        flags.DEFINE_string,
        None,
        "Where predicted structures and Foldseek alignments are cached. Defaults "
        "to a 'structural_templates' subdirectory of --output_dir.",
    )
    _define_once(
        "esmfold_model_dir",
        flags.DEFINE_string,
        None,
        "Local directory holding the ESMFold checkpoint (for example a download "
        "of facebook/esmfold_v1). Nothing is fetched over the network.",
    )
    _define_once(
        "esmfold_device",
        flags.DEFINE_string,
        "cuda",
        "Torch device for ESMFold, e.g. 'cuda' or 'cpu'.",
    )
    _define_once(
        "esmfold_chunk_size",
        flags.DEFINE_integer,
        None,
        "Chunk the ESMFold trunk to trade speed for GPU memory on long chains.",
    )
    _define_once(
        "esmfold_max_sequence_length",
        flags.DEFINE_integer,
        1500,
        "Refuse to fold sequences longer than this rather than risk an "
        "out-of-memory kill.",
    )


def structural_templates_enabled(flag_values: flags.FlagValues) -> bool:
    return bool(getattr(flag_values, "use_foldseek_templates", False))


def validate_structural_template_flags(flag_values: flags.FlagValues) -> None:
    """Fail early, and in one message, when the feature is misconfigured."""
    if not structural_templates_enabled(flag_values):
        return
    missing = [
        name for name in REQUIRED_WHEN_ENABLED if not getattr(flag_values, name, None)
    ]
    if missing:
        raise ValueError(
            "--use_foldseek_templates needs "
            + ", ".join(f"--{name}" for name in missing)
            + ". Structural template search cannot guess a database or a "
            "checkpoint location."
        )
    if getattr(flag_values, "use_mmseqs2", False):
        # The remote MMseqs2 path receives MSAs and templates together and has no
        # separable template search to replace.
        raise ValueError(
            "--use_foldseek_templates cannot be combined with --use_mmseqs2: that "
            "path obtains templates from the remote server, not from a local "
            "template search."
        )
    if getattr(flag_values, "data_pipeline", "alphafold2") != "alphafold2":
        raise ValueError(
            "--use_foldseek_templates applies to the AlphaFold 2 data pipeline. "
            "AlphaFold 3 runs its own template search internally."
        )


def structural_template_metadata_flags(
    flag_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop the structural-template flags from metadata unless they were used.

    Feature metadata is a record of what actually ran. Reporting a Foldseek
    binary that happened to be on PATH would be wrong, and it would also change
    the bytes of every feature file produced by a default run.
    """
    filtered = dict(flag_dict)
    if filtered.get("use_foldseek_templates"):
        return filtered
    for name in STRUCTURAL_TEMPLATE_FLAG_NAMES:
        filtered.pop(name, None)
    return filtered


def structure_database_spec(flag_values: flags.FlagValues) -> StructureDatabaseSpec:
    return StructureDatabaseSpec(
        name="foldseek",
        path=Path(flag_values.foldseek_database_path),
        identifier=flag_values.foldseek_database_id,
    )


def cache_dir(flag_values: flags.FlagValues, *, output_dir: str | Path | None) -> Path:
    configured = getattr(flag_values, "structural_template_cache_dir", None)
    if configured:
        return Path(configured)
    if not output_dir:
        raise ValueError(
            "--structural_template_cache_dir is required when there is no "
            "--output_dir to derive it from."
        )
    return Path(output_dir) / "structural_templates"


def foldseek_search_settings(
    flag_values: flags.FlagValues, *, output_dir: str | Path | None = None
) -> FoldseekSearchSettings:
    resolved_cache_dir = cache_dir(flag_values, output_dir=output_dir)
    temp_dir = flag_values.foldseek_temp_dir or resolved_cache_dir / "tmp"
    return FoldseekSearchSettings(
        database=structure_database_spec(flag_values),
        temp_dir=Path(temp_dir),
        cache_dir=resolved_cache_dir,
        e_value=flag_values.foldseek_e_value,
        max_hits=flag_values.foldseek_max_hits,
        min_alignment_tm_score=flag_values.foldseek_min_alignment_tm_score,
        alignment_type=flag_values.foldseek_alignment_type,
        threads=flag_values.foldseek_threads,
    )


def esmfold_settings(flag_values: flags.FlagValues) -> EsmfoldSettings:
    return EsmfoldSettings(
        model_dir=Path(flag_values.esmfold_model_dir),
        device=flag_values.esmfold_device,
        chunk_size=flag_values.esmfold_chunk_size,
        max_sequence_length=flag_values.esmfold_max_sequence_length,
    )


def build_structure_cache(
    flag_values: flags.FlagValues, *, output_dir: str | Path | None = None
) -> PredictedStructureCache:
    """The ESMFold stage on its own, for running structure prediction ahead."""
    return PredictedStructureCache(
        cache_dir=cache_dir(flag_values, output_dir=output_dir),
        predictor=EsmfoldStructurePredictor(esmfold_settings(flag_values)),
    )


def build_foldseek_template_searcher(
    flag_values: flags.FlagValues, *, output_dir: str | Path | None = None
) -> FoldseekTemplateSearcher:
    """The configured searcher, ready to stand in for hmmsearch or HHsearch."""
    validate_structural_template_flags(flag_values)
    return FoldseekTemplateSearcher(
        settings=foldseek_search_settings(flag_values, output_dir=output_dir),
        structures=build_structure_cache(flag_values, output_dir=output_dir),
        foldseek_process=SubprocessFoldseekProcess(flag_values.foldseek_binary_path),
    )
