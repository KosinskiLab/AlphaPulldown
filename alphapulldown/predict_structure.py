#
# This script is
# based on run_alphafold.py by DeepMind from https://github.com/deepmind/alphafold
# and contains code copied from the script run_alphafold.py.
# #
import json
import os
import pickle,gzip
import time
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax
import numpy as np
from alphapulldown.utils import get_run_alphafold
import jax.numpy as jnp


run_af = get_run_alphafold()
RELAX_MAX_ITERATIONS = run_af.RELAX_MAX_ITERATIONS
RELAX_ENERGY_TOLERANCE = run_af.RELAX_ENERGY_TOLERANCE
RELAX_STIFFNESS = run_af.RELAX_STIFFNESS
RELAX_EXCLUDE_RESIDUES = run_af.RELAX_EXCLUDE_RESIDUES
RELAX_MAX_OUTER_ITERATIONS = run_af.RELAX_MAX_OUTER_ITERATIONS

ModelsToRelax = run_af.ModelsToRelax

def _jnp_to_np(output):
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output

def get_score_from_result_pkl(pkl_path):
    """Get the score from the model result pkl file"""

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    if "iptm" in result:
        score_type = "iptm+ptm"
        score = 0.8 * result["iptm"] + 0.2 * result["ptm"]
    else:
        score_type = "plddt"
        score = np.mean(result["plddt"])

    return score_type, score

def get_score_from_result_pkl_gz(pkl_path):
    """Get the score from the model result pkl file"""

    with gzip.open(pkl_path, "rb") as f:
        result = pickle.load(f)
    if "iptm" in result:
        score_type = "iptm+ptm"
        score = 0.8 * result["iptm"] + 0.2 * result["ptm"]
    else:
        score_type = "plddt"
        score = np.mean(result["plddt"])

    return score_type, score

def get_existing_model_info(output_dir, model_runners):
    ranking_confidences = {}
    unrelaxed_proteins = {}
    unrelaxed_pdbs = {}
    processed_models = 0
    ranking_debug_path = os.path.join(output_dir, "ranking_debug.json")
    ranking_debug_temp_path = os.path.join(output_dir, "ranking_debug_temp.json")

    # Attempt to load existing processed model info from ranking_debug.json
    if os.path.exists(ranking_debug_path):
        with open(ranking_debug_path, "r") as file:
            ranking_data = json.load(file)
            # Extract the 'plddts' scores directly if available
            ranking_confidences = ranking_data.get("plddts", {})
            # Use the 'order' list to determine the count of processed models
            processed_models = len(ranking_data.get("order", []))
    # Fallback to the temp file if the main ranking_debug.json doesn't exist
    elif os.path.exists(ranking_debug_temp_path):
        with open(ranking_debug_temp_path, "r") as file:
            temp_data = json.load(file)
            ranking_confidences = temp_data.get("ranking_confidences", {})
            processed_models = len(ranking_confidences)

    for model_name, _ in model_runners.items():
        if model_name in ranking_confidences:
            # Skip processing of models already listed in ranking_confidences
            continue

        pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
        pkl_path = os.path.join(output_dir, f"result_{model_name}.pkl")
        pkl_gz_path = os.path.join(output_dir, f"result_{model_name}.pkl.gz")

        # Process .pkl or .pkl.gz files if they exist and haven't been processed
        score = None
        if os.path.exists(pkl_path) or os.path.exists(pkl_gz_path):
            try:
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        pickle.load(f)
                    score_name, score = get_score_from_result_pkl(pkl_path)
                elif os.path.exists(pkl_gz_path):
                    with gzip.open(pkl_gz_path, "rb") as f:
                        pickle.load(f)
                    score_name, score = get_score_from_result_pkl_gz(pkl_gz_path)
            except (EOFError, pickle.UnpicklingError):
                continue  # Skip files that cause errors

        if score is not None:
            ranking_confidences[model_name] = score
            # Update the temp ranking data after processing each model
            with open(ranking_debug_temp_path, "w") as temp_file:
                json.dump({"ranking_confidences": ranking_confidences}, temp_file)

        # Read unrelaxed pdb files if present
        if os.path.exists(pdb_path):
            with open(pdb_path, "r") as f:
                unrelaxed_pdb_str = f.read()
            unrelaxed_proteins[model_name] = protein.from_pdb_string(unrelaxed_pdb_str)
            unrelaxed_pdbs[model_name] = unrelaxed_pdb_str
            processed_models += 1

    return ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, processed_models

def predict(
    model_runners,
    output_dir,
    feature_dict,
    random_seed,
    benchmark,
    models_to_relax: ModelsToRelax,
    fasta_name,
    allow_resume=True,
    seqs=[],
    use_gpu_relax=True
):
    timings = {}
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}
    unrelaxed_proteins = {}
    prediction_result = {}
    START = 0
    ranking_output_path = os.path.join(output_dir, "ranking_debug.json")
    temp_timings_output_path = os.path.join(output_dir, "timings_temp.json") #To keep track of timings in case of crash and resume

    if allow_resume:
        logging.info("Checking for existing results")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START = get_existing_model_info(output_dir, model_runners)
        if os.path.exists(ranking_output_path) and len(unrelaxed_pdbs) == len(model_runners):
            logging.info(
                "ranking_debug.json exists. Skipping prediction. Restoring unrelaxed predictions and ranked order"
            )
            START = len(model_runners)
        elif START > 0:
            logging.info("Found existing results, continuing from there.")

    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        if model_index < START:
            continue
        logging.info("Running model %s on %s", model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed
        )
        timings[f"process_features_{model_name}"] = time.time() - t_0
        # Die if --multimeric_mode=True but no non-zero templates are in the feature dict
        if run_af.flags.FLAGS.multimeric_mode:
            if 'template_all_atom_positions' in processed_feature_dict:
                if np.any(processed_feature_dict['template_all_atom_positions']):
                    logging.info("Valid templates found with non-zero positions.")
                else:
                    raise ValueError("No valid templates found: all positions are zero.")
            else:
                raise ValueError("No template_all_atom_positions key found in processed_feature_dict.")
        t_0 = time.time()
        prediction_result = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed
        )

        # update prediction_result with input seqs
        prediction_result.update({"seqs": seqs})

        t_diff = time.time() - t_0
        timings[f"predict_and_compile_{model_name}"] = t_diff
        logging.info(
            "Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs",
            model_name,
            fasta_name,
            t_diff,
        )

        if benchmark:
            t_0 = time.time()
            model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed
            )
            t_diff = time.time() - t_0
            timings[f"predict_benchmark_{model_name}"] = t_diff
            logging.info(
                "Total JAX model %s on %s predict time (excludes compilation time): %.1fs",
                model_name,
                fasta_name,
                t_diff,
            )

        plddt = prediction_result["plddt"]
        ranking_confidences[model_name] = prediction_result["ranking_confidence"]

        # Remove jax dependency from results.
        np_prediction_result = _jnp_to_np(dict(prediction_result))

        result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
        with open(result_output_path, "wb") as f:
            pickle.dump(np_prediction_result, f, protocol=4)

        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1
        )
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode,
        )

        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdbs[model_name])

        with open(temp_timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))

    # Rank by model confidence.
    ranked_order = [
        model_name for model_name, confidence in
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

    # Relax predictions.
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=use_gpu_relax)

    if models_to_relax == ModelsToRelax.BEST:
        to_relax = [ranked_order[0]]
    elif models_to_relax == ModelsToRelax.ALL:
        to_relax = ranked_order
    elif models_to_relax == ModelsToRelax.NONE:
        to_relax = []

    for model_name in to_relax:
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[model_name])
        relax_metrics[model_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

    # Write out relaxed PDBs in rank order.
    for idx, model_name in enumerate(ranked_order):
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
            if model_name in relaxed_pdbs:
                model = relaxed_pdbs[model_name]
            else:
                model = unrelaxed_pdbs[model_name]
            f.write(model)
        # Calculate and report RMSD if multimeric templates were used.
        if run_af.flags.FLAGS.multimeric_mode:
            # Calculate RMSD
            from alphapulldown.analysis_pipeline.calculate_rmsd import calculate_rmsd_and_superpose
            import tempfile
            template_positions = feature_dict['template_all_atom_positions'][0]
            template_mask = feature_dict['template_all_atom_mask'][0]
            template_aatype = feature_dict['template_aatype'][0]
            template_residue_index = feature_dict.get('residue_index', None)
            chain_index = feature_dict['asym_id']
            b_factors = np.zeros(template_mask.shape, dtype=float)  # stub
            #template_sequence = feature_dict['template_sequence'] # doesn't work because we can't store b-strings there

            # Create the Protein object
            template_protein = protein.Protein(
                atom_positions=template_positions,
                atom_mask=template_mask,
                aatype=template_aatype,
                residue_index=template_residue_index,
                chain_index=chain_index,
                b_factors=b_factors
            )

            # Convert to PDB format
            pdb_string = protein.to_pdb(template_protein)

            with tempfile.TemporaryDirectory() as temp_dir:
                # TODO: do not geenrate a new template each time
                template_file_path = f"{temp_dir}/template.pdb"
                with open(template_file_path, 'w') as file:
                    file.write(pdb_string)
                # TODO: use template_sequence for alignment
                calculate_rmsd_and_superpose(template_file_path, ranked_output_path, temp_dir)

    if not os.path.exists(ranking_output_path):  # already exists if restored.
        with open(ranking_output_path, "w") as f:
            label = "iptm+ptm" if "iptm" in prediction_result else "plddts"
            f.write(
                json.dumps(
                    {label: ranking_confidences, "order": ranked_order}, indent=4
                )
            )

    logging.info("Final timings for %s: %s", fasta_name, timings)
    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    if models_to_relax != ModelsToRelax.NONE:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
            f.write(json.dumps(relax_metrics, indent=4))

    if os.path.exists(temp_timings_output_path): #should not happen at this stage but just in case
        try:
            os.remove(temp_timings_output_path)
        except OSError:
            pass
