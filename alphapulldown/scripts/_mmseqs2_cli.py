"""Shared lightweight flag schema for local MMseqs2 feature entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from absl import flags

from alphapulldown.feature_batch import (
    DATABASE_NAMES,
    DEFAULT_MAX_SEQUENCES,
    PAIRED_DATABASE_NAME,
    UNPAIRED_DATABASE_NAMES,
    DatabaseSelection,
    DatabaseSpec,
)


def _define_once(name: str, define: Callable, default, help_text: str) -> None:
    if name not in flags.FLAGS:
        define(name, default, help_text)


def define_msa_search_flags(
    *, include_fasta_paths: bool = False, include_summary_path: bool = False
) -> None:
    """Register the shared MSA-stage flags without duplicating global names."""
    if include_fasta_paths:
        _define_once(
            "fasta_paths",
            flags.DEFINE_list,
            None,
            "Paths to protein FASTA files.",
        )
    _define_once(
        "msa_output_dir",
        flags.DEFINE_string,
        None,
        "Durable per-protein MSA bundle directory.",
    )
    if include_summary_path:
        _define_once(
            "summary_path",
            flags.DEFINE_string,
            None,
            "Atomic whole-shard completion record.",
        )
    _define_once(
        "mmseqs_binary_path",
        flags.DEFINE_string,
        "/opt/mmseqs/bin/mmseqs",
        "Path to the bundled GPU-capable MMseqs2 executable.",
    )
    _define_once(
        "mmseqs_temp_dir",
        flags.DEFINE_string,
        None,
        "Fast local MMseqs2 scratch directory.",
    )
    _define_once(
        "mmseqs_batch_max_sequences",
        flags.DEFINE_integer,
        None,
        "Maximum unique sequences per query database.",
    )
    _define_once(
        "mmseqs_batch_max_residues",
        flags.DEFINE_integer,
        None,
        "Maximum residues per query database.",
    )
    _define_once(
        "mmseqs_e_value",
        flags.DEFINE_float,
        1e-4,
        "MMseqs2 search E-value cutoff.",
    )
    _define_once(
        "mmseqs_split_memory_limit",
        flags.DEFINE_string,
        None,
        "Memory MMseqs2 may use before splitting the target database, e.g. '150G'. "
        "Leave unset and MMseqs2 assumes 90% of the physical node memory, which "
        "ignores any cgroup limit a batch scheduler applied.",
    )
    _define_once(
        "mmseqs_use_gpu",
        flags.DEFINE_bool,
        True,
        "Run the MMseqs2 search on a GPU (requires GPU-padded databases and a GPU "
        "allocation). Set false to search on CPU instead.",
    )
    _define_once(
        "mmseqs_threads",
        flags.DEFINE_integer,
        8,
        "CPU threads for MMseqs2 operations.",
    )
    for database_name in DATABASE_NAMES:
        _define_once(
            f"mmseqs_{database_name}_database_path",
            flags.DEFINE_string,
            None,
            f"Explicit GPU-compatible MMseqs2 {database_name} database prefix.",
        )
        _define_once(
            f"mmseqs_{database_name}_database_id",
            flags.DEFINE_string,
            None,
            f"Immutable identifier for the {database_name} database build.",
        )
        _define_once(
            f"mmseqs_{database_name}_max_sequences",
            flags.DEFINE_integer,
            DEFAULT_MAX_SEQUENCES[database_name],
            f"Maximum {database_name} hits per query.",
        )


def define_template_provenance_flags() -> None:
    _define_once(
        "template_seqres_database_id",
        flags.DEFINE_string,
        None,
        "Immutable identity of the PDB seqres database used for templates.",
    )
    _define_once(
        "template_mmcif_database_id",
        flags.DEFINE_string,
        None,
        "Immutable identity of the mmCIF directory used for templates.",
    )


def database_spec(flag_values: flags.FlagValues, name: str) -> DatabaseSpec:
    return DatabaseSpec(
        name=name,
        path=Path(flag_values[f"mmseqs_{name}_database_path"].value),
        identifier=flag_values[f"mmseqs_{name}_database_id"].value,
        max_sequences=flag_values[f"mmseqs_{name}_max_sequences"].value,
    )


def database_selection(flag_values: flags.FlagValues) -> DatabaseSelection:
    """Configured databases with their roles named, rather than sliced by position."""
    return DatabaseSelection(
        unpaired=tuple(
            database_spec(flag_values, name) for name in UNPAIRED_DATABASE_NAMES
        ),
        paired=database_spec(flag_values, PAIRED_DATABASE_NAME),
    )


def required_msa_flag_names() -> tuple[str, ...]:
    return (
        "msa_output_dir",
        "mmseqs_binary_path",
        "mmseqs_temp_dir",
        "mmseqs_batch_max_sequences",
        "mmseqs_batch_max_residues",
        *(
            f"mmseqs_{database_name}_{suffix}"
            for database_name in DATABASE_NAMES
            for suffix in ("database_path", "database_id")
        ),
    )


def required_template_flag_names() -> tuple[str, ...]:
    return ("template_seqres_database_id", "template_mmcif_database_id")
