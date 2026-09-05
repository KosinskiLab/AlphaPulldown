#!/usr/bin/env python
"""Run independent AlphaPulldown folds through one resident model session."""

from absl import app, flags, logging

from alphapulldown.prediction_batch import (
    AlphaPulldownPredictionAdapter,
    execute_prediction_manifest,
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
    outcome = execute_prediction_manifest(
        FLAGS.manifest,
        AlphaPulldownPredictionAdapter(FLAGS, backend=prediction_command.backend),
    )

    if outcome.rejection is not None:
        logging.error("Prediction batch rejected: %s", outcome.rejection)
    if outcome.summary is not None:
        for failure in outcome.summary.failures:
            logging.error(
                "Prediction job %s failed: %s", failure.job_id, failure.message
            )
    logging.info(outcome.summary_message)
    if outcome.exit_code:
        raise SystemExit(outcome.exit_code)


if __name__ == "__main__":
    flags.mark_flag_as_required("manifest")
    flags.mark_flag_as_required("data_directory")
    flags.mark_flag_as_required("features_directory")
    app.run(main)
