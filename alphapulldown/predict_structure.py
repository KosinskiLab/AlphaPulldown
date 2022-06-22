#
# Authorship belongs to DeepMind
# based on run_alphafold.py from https://github.com/deepmind/alphafold
# #
import json
import os
import pickle
import time
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
import numpy as np
import json

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def predict(
    model_runners,
    output_dir,
    feature_dict,
    random_seed,
    benchmark,
    amber_relaxer,
    fasta_name,
    allow_resume=True,
    seqs=[],
):
    timings = {}
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    ranking_confidences = {}
    unrelaxed_proteins = {}
    if allow_resume:
        logging.info("Checking for %s", os.path.join(output_dir, "ranking_debug.json"))
    if (
        not os.path.exists(os.path.join(output_dir, "ranking_debug.json"))
        or not allow_resume
    ):
        # Run the models.
        num_models = len(model_runners)
        for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
            logging.info("Running model %s on %s", model_name, fasta_name)
            t_0 = time.time()
            model_random_seed = model_index + random_seed * num_models
            processed_feature_dict = model_runner.process_features(
                feature_dict, random_seed=model_random_seed
            )
            timings[f"process_features_{model_name}"] = time.time() - t_0

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

            result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
            with open(result_output_path, "wb") as f:
                pickle.dump(prediction_result, f, protocol=4)

            plddt_b_factors = np.repeat(
                plddt[:, None], residue_constants.atom_type_num, axis=-1
            )
            unrelaxed_protein = protein.from_prediction(
                features=processed_feature_dict,
                result=prediction_result,
                b_factors=plddt_b_factors,
                remove_leading_feature_dimension=not model_runner.multimer_mode,
            )

            unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
            unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
            with open(unrelaxed_pdb_path, "w") as f:
                f.write(unrelaxed_pdbs[model_name])

            if amber_relaxer:
                # Relax the prediction.
                t_0 = time.time()
                relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
                timings[f"relax_{model_name}"] = time.time() - t_0

                relaxed_pdbs[model_name] = relaxed_pdb_str

                # Save the relaxed PDB.
                relaxed_output_path = os.path.join(
                    output_dir, f"relaxed_{model_name}.pdb"
                )
                with open(relaxed_output_path, "w") as f:
                    f.write(relaxed_pdb_str)
        restored = False
    else:
        logging.info(
            "ranking_debug.json exists. Skipped prediction. Restoring unrelaxed predictions and ranked order"
        )
        ranking_file = open(os.path.join(output_dir, "ranking_debug.json"))
        ranking_confidences_data = json.load(ranking_file)
        ranking_confidences = ranking_confidences_data[
            list(ranking_confidences_data.keys())[0]
        ]
        for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
            unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
            logging.info("Restored pdb %s ", unrelaxed_pdb_path)
            logging.info(
                "Restored confidence for model %s: %s",
                model_name,
                ranking_confidences[model_name],
            )
            with open(unrelaxed_pdb_path, "r") as f:
                unrelaxed_pdb_str = f.read()
            unrelaxed_proteins[model_name] = protein.from_pdb_string(unrelaxed_pdb_str)
            unrelaxed_pdbs[model_name] = unrelaxed_pdb_str
        logging.info("Finished restoring unrelaxed PDBs.")
        restored = True
        if amber_relaxer:
            for model_index, (model_name, model_runner) in enumerate(
                model_runners.items()
            ):
                # Relax the prediction.
                t_0 = time.time()
                relaxed_pdb_str, _, _ = amber_relaxer.process(
                    prot=unrelaxed_proteins[model_name]
                )
                timings[f"relax_{model_name}"] = time.time() - t_0

                relaxed_pdbs[model_name] = relaxed_pdb_str

                # Save the relaxed PDB.
                relaxed_output_path = os.path.join(
                    output_dir, f"relaxed_{model_name}.pdb"
                )
                with open(relaxed_output_path, "w") as f:
                    f.write(relaxed_pdb_str)
    # Rank by model confidence and write out relaxed PDBs in rank order.
    ranked_order = []
    # pDockq_scores = {}
    for idx, (model_name, _) in enumerate(
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)
    ):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f"ranked_{idx}.pdb")
        with open(ranked_output_path, "w") as f:
            if amber_relaxer:
                f.write(relaxed_pdbs[model_name])
            else:
                f.write(unrelaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, "ranking_debug.json")
    if not restored:  # already exists if restored.
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
