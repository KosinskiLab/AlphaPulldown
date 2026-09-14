"""Building the AlphaFold 3 data pipeline from explicit settings.

The batch feature scripts used to reach into ``create_individual_features`` and call
``create_arguments()``, whose job is to *mutate the module-global FLAGS* -- filling in
database paths from ``--data_dir`` -- and then call ``create_pipeline_af3()`` and
``get_af3_feature_metadata()``, which read that mutation back. The interface between
the scripts was therefore the shape of a global, and correct only if the three calls
happened in that order.

This module states the same thing as data. ``AF3PipelineSettings.from_flags`` resolves
every path once and returns them; nothing it touches is written back. The settings also
carry ``resolved_flag_values`` so provenance metadata can record the paths that were
actually used without a global having been rewritten to hold them.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Callable, Dict, Mapping, Optional

# Binaries the AlphaFold 3 pipeline shells out to. Same name on the flags and on the
# AlphaFold 3 config, so one tuple describes both sides.
BINARY_FIELDS = (
    "jackhmmer_binary_path",
    "nhmmer_binary_path",
    "hmmalign_binary_path",
    "hmmsearch_binary_path",
    "hmmbuild_binary_path",
)

# AlphaFold 3 config field -> (flag name, database key). The three names differ often
# enough -- uniprot_cluster_annot_database_path / uniprot_database_path / uniprot -- that
# writing them out once is what keeps a typo from silently selecting the wrong database.
DATABASE_FIELDS: Mapping[str, tuple[str, str]] = {
    "small_bfd_database_path": ("small_bfd_database_path", "small_bfd"),
    "mgnify_database_path": ("mgnify_database_path", "mgnify"),
    "uniprot_cluster_annot_database_path": ("uniprot_database_path", "uniprot"),
    "uniref90_database_path": ("uniref90_database_path", "uniref90"),
    "ntrna_database_path": ("ntrna_database_path", "ntrna"),
    "rfam_database_path": ("rfam_database_path", "rfam"),
    "rna_central_database_path": ("rna_central_database_path", "rna_central"),
    "pdb_database_path": ("template_mmcif_dir", "template_mmcif_dir"),
    "seqres_database_path": ("pdb_seqres_database_path", "pdb_seqres"),
}

# Number of CPUs the two search stages get. Previously inline literals.
JACKHMMER_N_CPU = 8
NHMMER_N_CPU = 8


@dataclasses.dataclass(frozen=True)
class AF3PipelineSettings:
    """Everything the AlphaFold 3 data pipeline needs, resolved and explicit."""

    max_template_date: str
    binaries: Mapping[str, Optional[str]]
    # Keyed by AlphaFold 3 config field.
    databases: Mapping[str, Optional[str]]
    # Keyed by flag name, for provenance. Same values, addressed the way metadata
    # addresses them, so recording what was used needs no global to have been rewritten.
    resolved_flag_values: Mapping[str, Optional[str]]

    @classmethod
    def from_flags(
        cls,
        flags: Any,
        resolve_database_path: Callable[[str], Optional[str]],
    ) -> "AF3PipelineSettings":
        """Resolve settings from a parsed invocation without writing anything back.

        ``resolve_database_path`` maps a database key to its default location under
        ``--data_dir``; it is injected so this module does not depend on the CLI it
        was extracted from. An explicitly given flag always wins over the default.
        """
        binaries = {
            field: getattr(flags, field, None) for field in BINARY_FIELDS
        }
        databases: Dict[str, Optional[str]] = {}
        resolved_flag_values: Dict[str, Optional[str]] = {}
        for config_field, (flag_name, database_key) in DATABASE_FIELDS.items():
            value = getattr(flags, flag_name, None) or resolve_database_path(database_key)
            databases[config_field] = value
            resolved_flag_values[flag_name] = value
        return cls(
            max_template_date=flags.max_template_date,
            binaries=binaries,
            databases=databases,
            resolved_flag_values=resolved_flag_values,
        )

    def flag_values_with_resolved_paths(
        self, flag_values: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """``flag_values`` with the database paths this run actually used.

        The explicit replacement for reading a FLAGS object that ``create_arguments``
        had mutated on the caller's behalf.
        """
        return {**flag_values, **self.resolved_flag_values}


def build_pipeline(
    settings: AF3PipelineSettings,
    *,
    pipeline_cls: Any = None,
    config_cls: Any = None,
) -> Any:
    """Construct the AlphaFold 3 data pipeline described by ``settings``.

    The classes are parameters so this can be exercised without AlphaFold 3 installed;
    left unset, the real ones are imported and their absence is reported the way the
    CLI has always reported it.
    """
    if pipeline_cls is None or config_cls is None:
        try:
            from alphafold3.data.pipeline import (
                DataPipeline as _Pipeline,
                DataPipelineConfig as _Config,
            )
        except ImportError as exc:
            raise ImportError(
                "AlphaFold3 is not installed correctly. "
                "Install AlphaPulldown with 'pip install -e \".[alphafold3,test]\"', "
                "make sure the build environment provides SQLite, then build the "
                "vendored package with 'pip install -r alphafold3/dev-requirements.txt', "
                "'pip install --no-deps -e ./alphafold3', and 'build_data'."
            ) from exc
        pipeline_cls = pipeline_cls or _Pipeline
        config_cls = config_cls or _Config

    config = config_cls(
        **settings.binaries,
        **settings.databases,
        jackhmmer_n_cpu=JACKHMMER_N_CPU,
        nhmmer_n_cpu=NHMMER_N_CPU,
        max_template_date=datetime.date.fromisoformat(settings.max_template_date),
    )
    return pipeline_cls(config)
