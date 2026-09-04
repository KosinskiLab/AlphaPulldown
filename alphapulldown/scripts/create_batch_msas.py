#!/usr/bin/env python3
"""GPU-only MMseqs2 MSA stage; intentionally imports neither AlphaFold nor JAX."""

from __future__ import annotations

from pathlib import Path

from absl import app, flags, logging

from alphapulldown.feature_batch import (
    MsaBatch,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
    protein_requests_from_fastas,
    write_batch_summary,
)
from alphapulldown.scripts._mmseqs2_cli import (
    database_selection,
    define_msa_search_flags,
    required_msa_flag_names,
)


define_msa_search_flags(include_fasta_paths=True, include_summary_path=True)

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv
    summary_path = Path(FLAGS.summary_path)
    summary_path.unlink(missing_ok=True)
    requests = protein_requests_from_fastas(FLAGS.fasta_paths)
    databases = database_selection(FLAGS)
    settings = MsaBatchSettings(
        output_dir=Path(FLAGS.msa_output_dir),
        temp_dir=Path(FLAGS.mmseqs_temp_dir),
        unpaired_databases=databases.unpaired,
        paired_database=databases.paired,
        max_sequences_per_batch=FLAGS.mmseqs_batch_max_sequences,
        max_residues_per_batch=FLAGS.mmseqs_batch_max_residues,
        threads=FLAGS.mmseqs_threads,
        e_value=FLAGS.mmseqs_e_value,
        split_memory_limit=FLAGS.mmseqs_split_memory_limit,
    )
    result = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(
            FLAGS.mmseqs_binary_path, gpu=FLAGS.mmseqs_use_gpu
        ),
    ).generate(requests)
    logging.info(
        "MMseqs2-GPU MSA stage: %d written, %d reused, %d failed, %d query-only",
        len(result.written),
        len(result.reused),
        len(result.failures),
        len(result.query_only),
    )
    produced = len(result.written) + len(result.reused)
    if produced and len(result.query_only) == produced:
        raise RuntimeError(
            "Every MSA in this shard contains only the query sequence. A whole shard of "
            "orphan proteins is not plausible; check that the configured MMseqs2 "
            f"databases exist and are searchable ({', '.join(result.query_only)})"
        )
    if result.failures:
        detail = ", ".join(
            f"{failure.name} ({failure.error})" for failure in result.failures
        )
        raise RuntimeError(
            f"MMseqs2-GPU MSA stage failed for {len(result.failures)} protein(s): {detail}"
        )
    write_batch_summary(summary_path, result)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "summary_path",
            *required_msa_flag_names(),
        ]
    )
    app.run(main)
