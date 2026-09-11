#!/usr/bin/env python3
"""Compatibility CLI composing MMseqs2-GPU MSA and AF3 finalization stages."""

from __future__ import annotations

import os

# The composed direct mode imports AF3. Prevent JAX from reserving MMseqs' GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path
from typing import Sequence

from absl import app, flags, logging

from alphapulldown.feature_batch import (
    PROTEIN,
    FeatureBatch,
    FeatureBatchSettings,
    FeatureRequest,
    SubprocessMmseqsProcess,
    feature_requests_from_fastas,
)
from alphapulldown.scripts import create_individual_features as legacy_features
from alphapulldown.scripts._mmseqs2_cli import (
    accepted_molecule_types,
    always_required_msa_flag_names,
    database_selection,
    define_msa_search_flags,
    define_template_provenance_flags,
    require_msa_flags,
    required_template_flag_names,
)


define_msa_search_flags()
define_template_provenance_flags()

FLAGS = flags.FLAGS


def _feature_requests(fasta_paths: Sequence[str]) -> tuple[FeatureRequest, ...]:
    return feature_requests_from_fastas(
        fasta_paths, molecule_types=accepted_molecule_types(FLAGS)
    )


def main(argv) -> None:
    del argv
    if FLAGS.data_pipeline != "alphafold3":
        raise ValueError(
            "Batched local MMseqs2-GPU features require --data_pipeline=alphafold3"
        )
    if FLAGS.use_mmseqs2 or FLAGS.keep_msas or FLAGS.skip_msa or FLAGS.path_to_mmt:
        raise ValueError(
            "Batched local MMseqs2-GPU generation cannot be combined with "
            "--use_mmseqs2, --keep_msas, --skip_msa, or --path_to_mmt"
        )

    # No create_arguments() here: it worked by mutating the global FLAGS and the two
    # calls below then read that mutation back, so this stage depended on call order
    # across a script boundary. The settings resolve the same paths explicitly.
    settings = legacy_features.af3_pipeline_settings()
    pipeline = legacy_features.create_pipeline_af3()
    requests = _feature_requests(FLAGS.fasta_paths)
    chain_kinds = {request.molecule_type for request in requests} or {PROTEIN}
    require_msa_flags(FLAGS, chain_kinds)
    metadata = legacy_features.get_af3_feature_metadata(
        chain_kinds,
        skip_msa=True,
        flag_values=settings.flag_values_with_resolved_paths(FLAGS.flag_values_dict()),
    )
    databases = database_selection(FLAGS)
    result = FeatureBatch(
        settings=FeatureBatchSettings(
            output_dir=Path(FLAGS.output_dir),
            msa_output_dir=Path(FLAGS.msa_output_dir),
            temp_dir=Path(FLAGS.mmseqs_temp_dir),
            unpaired_databases=databases.unpaired,
            paired_database=databases.paired,
            max_sequences_per_batch=FLAGS.mmseqs_batch_max_sequences,
            max_residues_per_batch=FLAGS.mmseqs_batch_max_residues,
            threads=FLAGS.mmseqs_threads,
            e_value=FLAGS.mmseqs_e_value,
            num_iterations=FLAGS.mmseqs_num_iterations,
            split_memory_limit=FLAGS.mmseqs_split_memory_limit,
            rna_databases=databases.rna,
            rna_e_value=FLAGS.mmseqs_rna_e_value,
            compress=FLAGS.compress_features,
            base_metadata=metadata,
            max_template_date=FLAGS.max_template_date,
            template_seqres_database_id=FLAGS.template_seqres_database_id,
            template_mmcif_database_id=FLAGS.template_mmcif_database_id,
        ),
        mmseqs_process=SubprocessMmseqsProcess(
            FLAGS.mmseqs_binary_path,
            gpu=FLAGS.mmseqs_use_gpu,
            db_load_mode=FLAGS.mmseqs_db_load_mode,
        ),
        af3_pipeline=pipeline,
    ).generate(requests)
    logging.info(
        "Batched local MMseqs2-GPU features: %d written, %d reused, %d failed",
        len(result.written),
        len(result.reused),
        len(result.failures),
    )
    produced = len(result.written) + len(result.reused)
    if produced and len(result.query_only) == produced:
        raise RuntimeError(
            "Every MSA in this batch contains only the query sequence; check that the "
            f"configured MMseqs2 databases exist and are searchable "
            f"({', '.join(result.query_only)})"
        )
    if result.failures:
        detail = ", ".join(f"{item.name} ({item.error})" for item in result.failures)
        raise RuntimeError(
            f"Failed to create {len(result.failures)} artifact(s): {detail}"
        )


if __name__ == "__main__":
    # The database flags a run needs depend on the molecule types in its FASTA, which
    # absl cannot see at parse time; require_msa_flags finishes the check in main().
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "data_dir",
            "output_dir",
            "max_template_date",
            *required_template_flag_names(),
            *always_required_msa_flag_names(),
        ]
    )
    app.run(main)
