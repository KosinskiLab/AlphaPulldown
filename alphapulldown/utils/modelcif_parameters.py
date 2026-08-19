"""Shared policy for safe command parameters exported to ModelCIF."""

from __future__ import annotations

import re


MODELCIF_IGNORED_PARAMETER_NAMES = frozenset(
    {
        "?",
        "alsologtostderr",
        "bfd_database_path",
        "compress_features",
        "data_dir",
        "data_pipeline",
        "delta_threshold",
        "description_file",
        "fasta_paths",
        "hbm_oom_exit",
        "hhblits_binary_path",
        "hhsearch_binary_path",
        "hmmalign_binary_path",
        "hmmbuild_binary_path",
        "hmmsearch_binary_path",
        "jackhmmer_binary_path",
        "kalign_binary_path",
        "log_dir",
        "logger_levels",
        "logtostderr",
        "mgnify_database_path",
        "multiple_mmts",
        "nhmmer_binary_path",
        "ntrna_database_path",
        "obsolete_pdbs_path",
        "only_check_args",
        "op_conversion_fallback_to_while_loop",
        "output_dir",
        "path_to_mmt",
        "pdb",
        "pdb70_database_path",
        "pdb_post_mortem",
        "pdb_seqres_database_path",
        "protein",
        "rfam_database_path",
        "rna_central_database_path",
        "run_with_pdb",
        "run_with_profiling",
        "runtime_oom_exit",
        "showprefixforinfo",
        "small_bfd_database_path",
        "stderrthreshold",
        "template_mmcif_dir",
        "tt_check_filter",
        "tt_single_core_summaries",
        "uniprot_database_path",
        "uniref30_database_path",
        "uniref90_database_path",
        "use_small_bfd",
        "v",
        "verbosity",
        "xml_output_file",
    }
)

_MODELCIF_INDEXED_INPUT_PARAMETER = re.compile(
    r"(?:fasta_paths|multimeric_chains|multimeric_templates|protein)_\d+"
)

_MODELCIF_LOCAL_PATH_SUFFIXES = (
    "_binary_path",
    "_database_path",
    "_dir",
    "_path",
)


def is_modelcif_parameter(name: str) -> bool:
    """Return whether a metadata flag is safe to expose as a ModelCIF parameter."""
    return (
        name not in MODELCIF_IGNORED_PARAMETER_NAMES
        and _MODELCIF_INDEXED_INPUT_PARAMETER.fullmatch(name) is None
        and not name.endswith(_MODELCIF_LOCAL_PATH_SUFFIXES)
    )
