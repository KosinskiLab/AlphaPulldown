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
    FeatureBatch,
    FeatureBatchSettings,
    FeatureRequest,
    SubprocessMmseqsProcess,
    protein_requests_from_fastas,
)
from alphapulldown.scripts import create_individual_features as legacy_features
from alphapulldown.scripts._mmseqs2_cli import (
    DATABASE_NAMES,
    database_spec,
    define_msa_search_flags,
    define_template_provenance_flags,
    required_msa_flag_names,
    required_template_flag_names,
)


define_msa_search_flags()
define_template_provenance_flags()

FLAGS = flags.FLAGS


def _feature_requests(fasta_paths: Sequence[str]) -> tuple[FeatureRequest, ...]:
    return protein_requests_from_fastas(fasta_paths)


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

    legacy_features.create_arguments()
    pipeline = legacy_features.create_pipeline_af3()
    metadata = legacy_features.get_af3_feature_metadata({"protein"}, skip_msa=True)
    requests = _feature_requests(FLAGS.fasta_paths)
    result = FeatureBatch(
        settings=FeatureBatchSettings(
            output_dir=Path(FLAGS.output_dir),
            msa_output_dir=Path(FLAGS.msa_output_dir),
            temp_dir=Path(FLAGS.mmseqs_temp_dir),
            unpaired_databases=tuple(
                database_spec(FLAGS, name) for name in DATABASE_NAMES[:3]
            ),
            paired_database=database_spec(FLAGS, "uniprot"),
            max_sequences_per_batch=FLAGS.mmseqs_batch_max_sequences,
            max_residues_per_batch=FLAGS.mmseqs_batch_max_residues,
            threads=FLAGS.mmseqs_threads,
            e_value=FLAGS.mmseqs_e_value,
            compress=FLAGS.compress_features,
            base_metadata=metadata,
            max_template_date=FLAGS.max_template_date,
            template_seqres_database_id=FLAGS.template_seqres_database_id,
            template_mmcif_database_id=FLAGS.template_mmcif_database_id,
        ),
        mmseqs_process=SubprocessMmseqsProcess(FLAGS.mmseqs_binary_path),
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
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "data_dir",
            "output_dir",
            "max_template_date",
            *required_template_flag_names(),
            *required_msa_flag_names(),
        ]
    )
    app.run(main)
