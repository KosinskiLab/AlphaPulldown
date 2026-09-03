#!/usr/bin/env python
"""Run independent AlphaPulldown folds through one resident model session."""

from absl import app, flags, logging

from alphapulldown.prediction_batch import (
    AlphaPulldownPredictionAdapter,
    PredictionBatch,
)
from alphapulldown.scripts import run_structure_prediction as prediction_command


flags.DEFINE_string(
    "manifest",
    None,
    "JSONL manifest containing independent prediction jobs.",
)

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv
    prediction_command._validate_flags_for_backend(FLAGS.fold_backend)
    batch = PredictionBatch.from_jsonl(FLAGS.manifest)
    summary = batch.run(
        AlphaPulldownPredictionAdapter(FLAGS, backend=prediction_command.backend)
    )

    logging.info(
        "Prediction batch summary: %d completed, %d failed",
        len(summary.completed_job_ids),
        len(summary.failures),
    )
    for failure in summary.failures:
        logging.error("Prediction job %s failed: %s", failure.job_id, failure.message)
    if summary.failures:
        raise SystemExit(summary.exit_code)


if __name__ == "__main__":
    flags.mark_flag_as_required("manifest")
    flags.mark_flag_as_required("data_directory")
    flags.mark_flag_as_required("features_directory")
    app.run(main)
