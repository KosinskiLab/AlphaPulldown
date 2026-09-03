#!/usr/bin/env python3
"""CPU-only native AF3 template search and feature finalization stage."""

from __future__ import annotations

import os

# AF3 imports JAX transitively; this CPU stage must never initialize a GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

from absl import app, flags, logging

from alphapulldown.feature_batch import (
    FeatureFinalizationSettings,
    FeatureFinalizer,
    protein_requests_from_fastas,
)
from alphapulldown.scripts import create_individual_features as legacy_features


flags.DEFINE_string("msa_input_dir", None, "Directory containing MMseqs2 MSA bundles.")
flags.DEFINE_string(
    "template_seqres_database_id",
    None,
    "Immutable identity of the PDB seqres database used for templates.",
)
flags.DEFINE_string(
    "template_mmcif_database_id",
    None,
    "Immutable identity of the mmCIF directory used for templates.",
)

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv
    if FLAGS.data_pipeline != "alphafold3":
        raise ValueError("MMseqs2 MSA finalization requires --data_pipeline=alphafold3")
    if FLAGS.keep_msas or FLAGS.skip_msa or FLAGS.path_to_mmt or FLAGS.use_mmseqs2:
        raise ValueError(
            "MMseqs2 MSA finalization cannot be combined with --keep_msas, "
            "--skip_msa, --path_to_mmt, or --use_mmseqs2"
        )

    legacy_features.create_arguments()
    pipeline = legacy_features.create_pipeline_af3()
    metadata = legacy_features.get_af3_feature_metadata({"protein"}, skip_msa=True)
    requests = protein_requests_from_fastas(FLAGS.fasta_paths)
    result = FeatureFinalizer(
        settings=FeatureFinalizationSettings(
            output_dir=Path(FLAGS.output_dir),
            msa_input_dir=Path(FLAGS.msa_input_dir),
            max_template_date=FLAGS.max_template_date,
            template_seqres_database_id=FLAGS.template_seqres_database_id,
            template_mmcif_database_id=FLAGS.template_mmcif_database_id,
            compress=FLAGS.compress_features,
            base_metadata=metadata,
        ),
        af3_pipeline=pipeline,
    ).generate(requests)
    logging.info(
        "AF3 feature finalization: %d written, %d reused, %d failed",
        len(result.written),
        len(result.reused),
        len(result.failures),
    )
    if result.failures:
        detail = ", ".join(
            f"{failure.name} ({failure.error})" for failure in result.failures
        )
        raise RuntimeError(
            f"AF3 feature finalization failed for {len(result.failures)} protein(s): {detail}"
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "msa_input_dir",
            "output_dir",
            "data_dir",
            "max_template_date",
            "template_seqres_database_id",
            "template_mmcif_database_id",
        ]
    )
    app.run(main)
