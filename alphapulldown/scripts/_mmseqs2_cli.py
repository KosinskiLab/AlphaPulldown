"""Shared lightweight flag schema for local MMseqs2 feature entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from absl import flags

from alphapulldown.feature_batch import (
    DATABASE_NAMES,
    DEFAULT_MAX_SEQUENCES,
    DEFAULT_RNA_E_VALUE,
    DEFAULT_RNA_MAX_SEQUENCES,
    PAIRED_DATABASE_NAME,
    PROTEIN,
    RNA,
    RNA_DATABASE_NAMES,
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
        "mmseqs_num_iterations",
        flags.DEFINE_integer,
        1,
        "Iterative profile search iterations, as jackhmmer's n_iter. AlphaFold 3 uses "
        "3; MMseqs2 defaults to 1, a plain sequence-sequence search. Raising it costs "
        "time and recovers sensitivity on shallow families.",
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
    _define_once(
        "mmseqs_rna_e_value",
        flags.DEFINE_float,
        DEFAULT_RNA_E_VALUE,
        "MMseqs2 search E-value cutoff for RNA chains. AlphaFold 3 searches its own "
        "RNA databases at 1e-3, which is the default here too.",
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
    for database_name in RNA_DATABASE_NAMES:
        _define_once(
            f"mmseqs_{database_name}_database_path",
            flags.DEFINE_string,
            None,
            f"Explicit MMseqs2 nucleotide {database_name} database prefix. Setting "
            "all three RNA database paths is what enables RNA chains; leave them "
            "unset and this stage stays protein-only.",
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
            DEFAULT_RNA_MAX_SEQUENCES[database_name],
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


def _flag_value(flag_values: flags.FlagValues, name: str):
    """Read a flag that a caller may not have defined at all."""
    try:
        flag = flag_values[name]
    except KeyError:
        return None
    return flag.value


def database_spec(
    flag_values: flags.FlagValues, name: str, molecule_type: str = PROTEIN
) -> DatabaseSpec:
    return DatabaseSpec(
        name=name,
        path=Path(flag_values[f"mmseqs_{name}_database_path"].value),
        identifier=flag_values[f"mmseqs_{name}_database_id"].value,
        max_sequences=flag_values[f"mmseqs_{name}_max_sequences"].value,
        molecule_type=molecule_type,
    )


def rna_database_specs(flag_values: flags.FlagValues) -> tuple[DatabaseSpec, ...]:
    """The RNA databases, or nothing at all when none of them is configured.

    Setting some but not all of them is a configuration mistake rather than a partial
    opt-in: AlphaFold 3 merges all three into one RNA MSA, so a subset would silently
    produce a shallower alignment than the same run on another machine.
    """
    configured = tuple(
        name
        for name in RNA_DATABASE_NAMES
        if _flag_value(flag_values, f"mmseqs_{name}_database_path")
    )
    if not configured:
        return ()
    missing = tuple(name for name in RNA_DATABASE_NAMES if name not in configured)
    if missing:
        raise ValueError(
            "RNA MSAs need all of "
            + ", ".join(RNA_DATABASE_NAMES)
            + "; missing "
            + ", ".join(f"--mmseqs_{name}_database_path" for name in missing)
        )
    return tuple(
        database_spec(flag_values, name, RNA) for name in RNA_DATABASE_NAMES
    )


def database_selection(flag_values: flags.FlagValues) -> DatabaseSelection:
    """Configured databases with their roles named, rather than sliced by position."""
    # An RNA-only shard never reaches the protein databases, so they may be unset. A
    # shard that does contain protein has already been checked by require_msa_flags.
    protein_configured = all(
        _flag_value(flag_values, f"mmseqs_{name}_database_path")
        for name in DATABASE_NAMES
    )
    return DatabaseSelection(
        unpaired=tuple(
            database_spec(flag_values, name) for name in UNPAIRED_DATABASE_NAMES
        )
        if protein_configured
        else (),
        paired=(
            database_spec(flag_values, PAIRED_DATABASE_NAME)
            if protein_configured
            else None
        ),
        rna=rna_database_specs(flag_values),
    )


def accepted_molecule_types(flag_values: flags.FlagValues) -> tuple[str, ...]:
    """Protein always; RNA once the RNA databases have been configured."""
    if rna_database_specs(flag_values):
        return (PROTEIN, RNA)
    return (PROTEIN,)


def required_msa_flag_names(
    molecule_types: Sequence[str] = (PROTEIN,),
) -> tuple[str, ...]:
    database_names: tuple[str, ...] = ()
    if PROTEIN in molecule_types:
        database_names += DATABASE_NAMES
    if RNA in molecule_types:
        database_names += RNA_DATABASE_NAMES
    return (
        "msa_output_dir",
        "mmseqs_binary_path",
        "mmseqs_temp_dir",
        "mmseqs_batch_max_sequences",
        "mmseqs_batch_max_residues",
        *(
            f"mmseqs_{database_name}_{suffix}"
            for database_name in database_names
            for suffix in ("database_path", "database_id")
        ),
    )


def require_msa_flags(
    flag_values: flags.FlagValues, molecule_types: Sequence[str]
) -> None:
    """Fail on unset flags for exactly the molecule types this shard contains.

    Which databases a shard needs depends on what is in its FASTA, which absl cannot
    know when flags are parsed - so an RNA-only shard is not made to point at the
    protein databases just to satisfy a static requirement.
    """
    missing = [
        name
        for name in required_msa_flag_names(molecule_types)
        if _flag_value(flag_values, name) is None
    ]
    if missing:
        raise flags.IllegalFlagValueError(
            "These flags must have a value: "
            + ", ".join(f"--{name}" for name in missing)
        )


def always_required_msa_flag_names() -> tuple[str, ...]:
    """The flags every shard needs, whatever molecule types it turns out to hold."""
    return required_msa_flag_names(molecule_types=())


def required_template_flag_names() -> tuple[str, ...]:
    return ("template_seqres_database_id", "template_mmcif_database_id")
