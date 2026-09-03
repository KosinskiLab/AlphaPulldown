#!/usr/bin/env python3
"""Thin command-line adapter for batched local MMseqs2-GPU AF3 features."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence

from absl import app, flags, logging

from alphapulldown.feature_batch import (
    DatabaseSpec,
    FeatureBatch,
    FeatureBatchSettings,
    FeatureRequest,
    SubprocessMmseqsProcess,
)
from alphapulldown.scripts import create_individual_features as legacy_features
from alphapulldown.utils.file_handling import iter_seqs


flags.DEFINE_string(
    "mmseqs_binary_path", shutil.which("mmseqs"), "Path to the MMseqs2 executable."
)
flags.DEFINE_string(
    "mmseqs_temp_dir", None, "Fast local directory for temporary MMseqs2 files."
)
flags.DEFINE_integer(
    "mmseqs_batch_max_sequences",
    None,
    "Maximum unique protein sequences in one MMseqs2 query database.",
)
flags.DEFINE_integer(
    "mmseqs_batch_max_residues",
    None,
    "Maximum total residues in one MMseqs2 query database.",
)
flags.DEFINE_float("mmseqs_sensitivity", 7.5, "MMseqs2 search sensitivity.")
flags.DEFINE_float("mmseqs_e_value", 1e-4, "MMseqs2 search E-value cutoff.")
flags.DEFINE_integer("mmseqs_threads", 8, "CPU threads for MMseqs2 operations.")

for database_name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
    flags.DEFINE_string(
        f"mmseqs_{database_name}_database_path",
        None,
        f"Explicit local MMseqs2 {database_name} database prefix.",
    )
    flags.DEFINE_string(
        f"mmseqs_{database_name}_database_id",
        None,
        f"Immutable identifier for the configured {database_name} database build.",
    )

flags.DEFINE_integer(
    "mmseqs_uniref90_max_sequences", 10_000, "Maximum UniRef90 hits per query."
)
flags.DEFINE_integer(
    "mmseqs_mgnify_max_sequences", 5_000, "Maximum MGnify hits per query."
)
flags.DEFINE_integer(
    "mmseqs_small_bfd_max_sequences", 5_000, "Maximum small-BFD hits per query."
)
flags.DEFINE_integer(
    "mmseqs_uniprot_max_sequences", 50_000, "Maximum paired UniProt hits per query."
)

FLAGS = flags.FLAGS


def _database(name: str) -> DatabaseSpec:
    return DatabaseSpec(
        name=name,
        path=Path(getattr(FLAGS, f"mmseqs_{name}_database_path")),
        identifier=getattr(FLAGS, f"mmseqs_{name}_database_id"),
        max_sequences=getattr(FLAGS, f"mmseqs_{name}_max_sequences"),
    )


def _feature_requests(fasta_paths: Sequence[str]) -> tuple[FeatureRequest, ...]:
    requests = []
    for sequence, description in iter_seqs(fasta_paths):
        chain_kind = legacy_features.get_af3_chain_kind(description, sequence)
        if chain_kind != "protein":
            raise ValueError(
                "Batched local MMseqs2-GPU features accept proteins only; "
                f"{description!r} is {chain_kind}"
            )
        requests.append(FeatureRequest(name=description, sequence=sequence))
    return tuple(requests)


def main(argv):
    del argv
    if FLAGS.data_pipeline != "alphafold3":
        raise ValueError(
            "Batched local MMseqs2-GPU features require "
            "--data_pipeline=alphafold3"
        )
    if FLAGS.use_mmseqs2:
        raise ValueError(
            "--use_mmseqs2 selects the existing remote AF2 path and cannot be combined "
            "with batched local MMseqs2-GPU generation"
        )
    if FLAGS.keep_msas or FLAGS.skip_msa or FLAGS.path_to_mmt:
        raise ValueError(
            "Batched local MMseqs2-GPU generation does not support --keep_msas, "
            "--skip_msa, or --path_to_mmt"
        )

    legacy_features.create_arguments()
    native_pipeline = legacy_features.create_pipeline_af3()
    metadata = legacy_features.get_af3_feature_metadata(
        {"protein"}, skip_msa=True
    )
    requests = _feature_requests(FLAGS.fasta_paths)
    settings = FeatureBatchSettings(
        output_dir=Path(FLAGS.output_dir),
        temp_dir=Path(FLAGS.mmseqs_temp_dir),
        unpaired_databases=(
            _database("uniref90"),
            _database("mgnify"),
            _database("small_bfd"),
        ),
        paired_database=_database("uniprot"),
        max_sequences_per_batch=FLAGS.mmseqs_batch_max_sequences,
        max_residues_per_batch=FLAGS.mmseqs_batch_max_residues,
        threads=FLAGS.mmseqs_threads,
        sensitivity=FLAGS.mmseqs_sensitivity,
        e_value=FLAGS.mmseqs_e_value,
        compress=FLAGS.compress_features,
        base_metadata=metadata,
    )
    result = FeatureBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(FLAGS.mmseqs_binary_path),
        af3_pipeline=native_pipeline,
    ).generate(requests)

    logging.info(
        "Batched local MMseqs2-GPU features: %d written, %d reused, %d failed",
        len(result.written),
        len(result.reused),
        len(result.failures),
    )
    if result.failures:
        summary = ", ".join(
            f"{failure.name} ({failure.error})" for failure in result.failures
        )
        raise RuntimeError(
            f"Failed to create {len(result.failures)} AF3 feature artifact(s): "
            f"{summary}"
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "data_dir",
            "output_dir",
            "max_template_date",
            "mmseqs_binary_path",
            "mmseqs_temp_dir",
            "mmseqs_batch_max_sequences",
            "mmseqs_batch_max_residues",
            *[
                f"mmseqs_{database_name}_{suffix}"
                for database_name in ("uniref90", "mgnify", "small_bfd", "uniprot")
                for suffix in ("database_path", "database_id")
            ],
        ]
    )
    app.run(main)
