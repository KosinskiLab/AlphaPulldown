#!/usr/bin/env python3
"""CPU-only template search and feature finalization for local MMseqs2 bundles.

Reads the MSA bundles the search stage wrote and publishes standard features for
either backend: AlphaFold 3 input JSON, or AlphaFold 2 MonomericObject pickles.
"""

from __future__ import annotations

import os

# AF3 imports JAX transitively, and so does alphapulldown.objects via ColabFold;
# this CPU stage must never initialize a GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

from absl import app, flags, logging

from alphapulldown.af2_feature_finalizer import (
    Af2FeatureFinalizationSettings,
    Af2FeatureFinalizer,
    build_af2_template_stack,
)
from alphapulldown.feature_batch import (
    MOLECULE_TYPES,
    PROTEIN,
    FeatureFinalizationSettings,
    FeatureFinalizer,
    feature_requests_from_fastas,
)
from alphapulldown.scripts import create_individual_features as legacy_features
from alphapulldown.scripts._mmseqs2_cli import (
    define_template_provenance_flags,
    required_template_flag_names,
)
from alphapulldown.utils import save_meta_data


flags.DEFINE_string("msa_input_dir", None, "Directory containing MMseqs2 MSA bundles.")
define_template_provenance_flags()

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv
    if FLAGS.keep_msas or FLAGS.skip_msa or FLAGS.path_to_mmt or FLAGS.use_mmseqs2:
        raise ValueError(
            "MMseqs2 MSA finalization cannot be combined with --keep_msas, "
            "--skip_msa, --path_to_mmt, or --use_mmseqs2"
        )
    if FLAGS.data_pipeline == "alphafold2":
        result = _finalize_alphafold2()
        backend = "AF2"
    else:
        result = _finalize_alphafold3()
        backend = "AF3"
    logging.info(
        "%s feature finalization: %d written, %d reused, %d failed",
        backend,
        len(result.written),
        len(result.reused),
        len(result.failures),
    )
    if result.failures:
        detail = ", ".join(
            f"{failure.name} ({failure.error})" for failure in result.failures
        )
        raise RuntimeError(
            f"{backend} feature finalization failed for {len(result.failures)} "
            f"chain(s): {detail}"
        )


def af2_template_metadata(stack) -> dict:
    """Provenance for the resources an AlphaFold 2 finalization actually uses.

    Not the run's whole flag set: the jackhmmer and HHblits binary flags default to
    whatever is on PATH, so recording them would claim this MSA came from tools
    that never touched it. The MSA's own provenance is the bundle's, added by the
    finalizer. Only the template stack, which does run here, is recorded.
    """
    used = {
        "template_mmcif_dir": stack.template_mmcif_dir,
        "obsolete_pdbs_path": stack.obsolete_pdbs_path,
        "kalign_binary_path": stack.kalign_binary_path,
        "max_template_date": stack.max_template_date,
        "data_pipeline": "alphafold2",
    }
    if stack.use_hhsearch:
        used["hhsearch_binary_path"] = stack.hhsearch_binary_path
        used["pdb70_database_path"] = stack.pdb70_database_path
    else:
        used["hmmsearch_binary_path"] = stack.hmmsearch_binary_path
        used["hmmbuild_binary_path"] = stack.hmmbuild_binary_path
        used["pdb_seqres_database_path"] = stack.pdb_seqres_database_path
    return save_meta_data.get_meta_dict(used)


def _finalize_alphafold2():
    # Explicit settings, as for AlphaFold 3 below: paths come from --data_dir
    # without create_arguments() rewriting the global FLAGS.
    stack = legacy_features.af2_template_stack_settings()
    template_searcher, template_featurizer = build_af2_template_stack(stack)
    # Every molecule type the search stage can produce is read, so an RNA chain
    # fails here by name, with the reason, rather than as a missing bundle.
    requests = feature_requests_from_fastas(
        FLAGS.fasta_paths, molecule_types=MOLECULE_TYPES
    )
    return Af2FeatureFinalizer(
        settings=Af2FeatureFinalizationSettings(
            output_dir=Path(FLAGS.output_dir),
            msa_input_dir=Path(FLAGS.msa_input_dir),
            max_template_date=FLAGS.max_template_date,
            template_seqres_database_id=FLAGS.template_seqres_database_id,
            template_mmcif_database_id=FLAGS.template_mmcif_database_id,
            template_searcher=stack.searcher_name,
            compress=FLAGS.compress_features,
            base_metadata=af2_template_metadata(stack),
        ),
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
    ).generate(requests)


def _finalize_alphafold3():
    # No create_arguments() here: it worked by mutating the global FLAGS and the two
    # calls below then read that mutation back, so this stage depended on call order
    # across a script boundary. The settings resolve the same paths explicitly.
    settings = legacy_features.af3_pipeline_settings()
    pipeline = legacy_features.create_pipeline_af3()
    # This stage reads whatever the MSA stage produced, so it accepts every molecule
    # type that stage can search; a chain whose MSA was never generated fails here
    # with the missing bundle named.
    requests = feature_requests_from_fastas(
        FLAGS.fasta_paths, molecule_types=MOLECULE_TYPES
    )
    chain_kinds = {request.molecule_type for request in requests} or {PROTEIN}
    metadata = legacy_features.get_af3_feature_metadata(
        chain_kinds,
        skip_msa=True,
        flag_values=settings.flag_values_with_resolved_paths(FLAGS.flag_values_dict()),
    )
    return FeatureFinalizer(
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


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "msa_input_dir",
            "output_dir",
            "data_dir",
            "max_template_date",
            *required_template_flag_names(),
        ]
    )
    app.run(main)
