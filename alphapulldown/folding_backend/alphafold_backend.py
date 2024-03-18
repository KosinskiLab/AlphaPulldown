""" Implements structure prediction backend using AlphaFold.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import time
import json
import pickle
import tempfile
from typing import Dict
from os.path import join, exists

import numpy as np
import jax.numpy as jnp
from alphapulldown.predict_structure import get_existing_model_info
from alphapulldown.objects import MultimericObject
from alphapulldown.utils.plotting import create_and_save_pae_plots
from alphapulldown.utils.post_modelling import post_prediction_process
from alphapulldown.utils.calculate_rmsd import calculate_rmsd_and_superpose

# Avoid module not found error by importing after AP
import run_alphafold
from run_alphafold import ModelsToRelax
from alphafold.relax import relax
from alphafold.common import protein, residue_constants

from .folding_backend import FoldingBackend


def _jnp_to_np(output):
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


class AlphaFoldBackend(FoldingBackend):
    """
    A backend to perform structure prediction using AlphaFold.
    """

    @staticmethod
    def setup(
        model_name: str,
        num_cycle: int,
        model_dir: str,
        num_multimer_predictions_per_model: int,
        msa_depth_scan=False,
        model_names_custom: str = None,
        msa_depth=None,
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures multiple AlphaFold model runners.

        Parameters
        ----------
        model_name : str
            The preset model configuration name.
        num_cycle : int
            The number of recycling iterations to be used in prediction.
        model_dir : str
            The directory containing model parameters.
        num_multimer_predictions_per_model : int
            The number of multimer predictions to perform for each model.
        msa_depth_scan : bool, optional
            Whether to adjust MSA depth logarithmically, default is False.
        model_names_custom : str, optional
            Comma-separated custom model names to use instead of the default preset,
            default is None.
        msa_depth : int or None, optional
            A specific MSA depth to use, default is None.
        **kwargs : dict
            Additional keyword arguments for model runner configuration.

        Returns
        -------
        Dict
            A dictionary containing the configured model runners.

        Raises
        ------
        Exception
            If provided custom model names are not part of the available models.
        """

        from alphafold.model import config
        from alphafold.model import data, model

        num_ensemble = 1
        model_runners = {}
        model_names = config.MODEL_PRESETS[model_name]

        if model_names_custom:
            model_names_custom = tuple(model_names_custom.split(","))
            if all(x in model_names for x in model_names_custom):
                model_names = model_names_custom
            else:
                raise Exception(
                    f"Provided model names {model_names_custom} not part of available {model_names}"
                )

        for model_name in model_names:
            model_config = config.model_config(model_name)
            model_config.model.num_ensemble_eval = num_ensemble
            model_config["model"].update({"num_recycle": num_cycle})

            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=model_dir
            )
            model_runner = model.RunModel(model_config, model_params)

            if msa_depth_scan or msa_depth:
                embeddings_and_evo = model_config["model"]["embeddings_and_evoformer"]
                num_msa = embeddings_and_evo["num_msa"]
                num_extra_msa = embeddings_and_evo["num_extra_msa"]

                msa_ranges = np.rint(
                    np.logspace(
                        np.log10(16),
                        np.log10(num_msa),
                        num_multimer_predictions_per_model,
                    )
                ).astype(int)

                extra_msa_ranges = np.rint(
                    np.logspace(
                        np.log10(32),
                        np.log10(num_extra_msa),
                        num_multimer_predictions_per_model,
                    )
                ).astype(int)

            for i in range(num_multimer_predictions_per_model):
                if msa_depth or msa_depth_scan:
                    if msa_depth:
                        num_msa = int(msa_depth)
                        # approx. 4x the number of msa, as in the AF2 config file
                        num_extra_msa = int(num_msa * 4)
                    elif msa_depth_scan:
                        num_msa = int(msa_ranges[i])
                        num_extra_msa = int(extra_msa_ranges[i])

                    # Conversion to int before because num_msa could be None
                    embeddings_and_evo.update(
                        {"num_msa": num_msa, "num_extra_msa": num_extra_msa}
                    )

                    model_runners[f"{model_name}_pred_{i}_msa_{num_msa}"] = model_runner
                else:
                    model_runners[f"{model_name}_pred_{i}"] = model_runner

        return {"model_runner": model_runners}

    @staticmethod
    def predict(
        model_runner: Dict,
        multimeric_object: MultimericObject,
        output_dir: Dict,
        models_to_relax: object = ModelsToRelax,
        random_seed: int = 42,
        allow_resume: bool = True,
        use_gpu_relax: bool = True,
        multimeric_mode: bool = False,
        **kwargs,
    ) -> None:
        """
        Executes structure predictions using configured AlphaFold models.

        Parameters
        ----------
        model_runner : Dict
            Configured model runners with model names as keys.
        multimeric_object : MultimericObject
            An object containing features of the multimeric proteins.
        output_dir : str
            The directory to save prediction results and PDB files.
        models_to_relax : object, optional
            Specifies which models' predictions to relax, defaults to ModelsToRelax enum.
        random_seed : int, optional
            A seed for random number generation to ensure reproducibility, default is 42.
        allow_resume : bool, optional
            If set to True, resumes prediction from partially completed runs, default is True.
        use_gpu_relax : bool, optional
            If set to True, utilizes GPU acceleration for the relaxation step, default is True.
        multimeric_mode : bool, optional
            Enables multimeric prediction mode, default is False.
        **kwargs : dict
            Additional keyword arguments for prediction.

        Raises
        ------
        ValueError
            If multimeric mode is enabled but no valid templates are found.
        """
        timings = {}
        unrelaxed_pdbs = {}
        relaxed_pdbs = {}
        relax_metrics = {}
        ranking_confidences = {}
        unrelaxed_proteins = {}
        prediction_result = {}
        START = 0

        ranking_output_path = join(output_dir, "ranking_debug.json")

        if allow_resume:
            (
                ranking_confidences,
                unrelaxed_proteins,
                unrelaxed_pdbs,
                START,
            ) = get_existing_model_info(output_dir, model_runner)

            if exists(ranking_output_path) and len(unrelaxed_pdbs) == len(model_runner):
                START = len(model_runner)

        num_models = len(model_runner)
        for model_index, (model_name, model_runner) in enumerate(model_runner.items()):
            if model_index < START:
                continue
            t_0 = time.time()

            model_random_seed = model_index + random_seed * num_models
            processed_feature_dict = model_runner.process_features(
                multimeric_object.feature_dict, random_seed=model_random_seed
            )
            timings[f"process_features_{model_name}"] = time.time() - t_0
            # Die if --multimeric_mode=True but no non-zero templates are in the feature dict
            if multimeric_mode:
                if "template_all_atom_positions" in processed_feature_dict:
                    if not np.any(
                        processed_feature_dict["template_all_atom_positions"]
                    ):
                        raise ValueError(
                            "No valid templates found: all positions are zero."
                        )
                else:
                    raise ValueError(
                        "No template_all_atom_positions key found in processed_feature_dict."
                    )

            t_0 = time.time()
            prediction_result = model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed
            )

            # update prediction_result with input seqs
            prediction_result.update({"seqs": multimeric_object.input_seqs})

            t_diff = time.time() - t_0
            timings[f"predict_and_compile_{model_name}"] = t_diff

            plddt = prediction_result["plddt"]
            ranking_confidences[model_name] = prediction_result["ranking_confidence"]

            # Remove jax dependency from results.
            np_prediction_result = _jnp_to_np(dict(prediction_result))

            result_output_path = join(output_dir, f"result_{model_name}.pkl")
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
            unrelaxed_pdb_path = join(output_dir, f"unrelaxed_{model_name}.pdb")
            with open(unrelaxed_pdb_path, "w") as f:
                f.write(unrelaxed_pdbs[model_name])

        # Rank by model confidence.
        ranked_order = [
            model_name
            for model_name, confidence in sorted(
                ranking_confidences.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Relax predictions.
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=run_alphafold.RELAX_MAX_ITERATIONS,
            tolerance=run_alphafold.RELAX_ENERGY_TOLERANCE,
            stiffness=run_alphafold.RELAX_STIFFNESS,
            exclude_residues=run_alphafold.RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=run_alphafold.RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=use_gpu_relax,
        )

        to_relax = []
        if models_to_relax == ModelsToRelax.BEST:
            to_relax = [ranked_order[0]]
        elif models_to_relax == ModelsToRelax.ALL:
            to_relax = ranked_order

        for model_name in to_relax:
            t_0 = time.time()
            relaxed_pdb_str, _, violations = amber_relaxer.process(
                prot=unrelaxed_proteins[model_name]
            )
            relax_metrics[model_name] = {
                "remaining_violations": violations,
                "remaining_violations_count": sum(violations),
            }
            timings[f"relax_{model_name}"] = time.time() - t_0

            relaxed_pdbs[model_name] = relaxed_pdb_str

            # Save the relaxed PDB.
            relaxed_output_path = join(output_dir, f"relaxed_{model_name}.pdb")
            with open(relaxed_output_path, "w") as f:
                f.write(relaxed_pdb_str)

        # Write out relaxed PDBs in rank order.
        for idx, model_name in enumerate(ranked_order):
            ranked_output_path = join(output_dir, f"ranked_{idx}.pdb")
            with open(ranked_output_path, "w") as f:
                model = relaxed_pdbs.get(model_name, None)
                if model is None:
                    model = unrelaxed_pdbs[model_name]
                f.write(model)

            if multimeric_mode:
                feature_dict = multimeric_object.feature_dict
                template_mask = feature_dict["template_all_atom_mask"][0]
                template_protein = protein.Protein(
                    atom_positions=feature_dict["template_all_atom_positions"][0],
                    atom_mask=template_mask,
                    aatype=feature_dict["template_aatype"][0],
                    residue_index=feature_dict.get("residue_index", None),
                    chain_index=feature_dict["asym_id"],
                    b_factors=np.zeros(template_mask.shape, dtype=float),
                )
                pdb_string = protein.to_pdb(template_protein)

                with tempfile.TemporaryDirectory() as temp_dir:
                    # TODO: do not generate a new template each time
                    template_file_path = f"{temp_dir}/template.pdb"
                    with open(template_file_path, "w") as file:
                        file.write(pdb_string)
                    # TODO: use template_sequence for alignment
                    calculate_rmsd_and_superpose(
                        template_file_path, ranked_output_path, temp_dir
                    )

            # already exists if restored
            if not exists(ranking_output_path):
                with open(ranking_output_path, "w") as f:
                    label = "iptm+ptm" if "iptm" in prediction_result else "plddts"
                    f.write(
                        json.dumps(
                            {label: ranking_confidences, "order": ranked_order},
                            indent=4,
                        )
                    )

        timings_output_path = join(output_dir, "timings.json")
        with open(timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))
        if models_to_relax != ModelsToRelax.NONE:
            relax_metrics_path = join(output_dir, "relax_metrics.json")
            with open(relax_metrics_path, "w") as f:
                f.write(json.dumps(relax_metrics, indent=4))

    @staticmethod
    def postprocess(
        multimeric_object: MultimericObject,
        output_dir: str,
        zip_pickles: bool = False,
        remove_pickles: bool = False,
        **kwargs: Dict,
    ) -> None:
        """
        Performs post-processing operations on predicted protein structures and
        writes results and plots to output_dir.

        Parameters
        ----------
        multimeric_object : MultimericObject
            The multimeric object containing the predicted structures and
            associated data.
        output_dir : str
            The directory where post-processed files and plots will be saved.
        zip_pickles : bool, optional
            If True, zips the pickle files containing prediction results.
            Default is False.
        remove_pickles : bool, optional
            If True, removes the pickle files after post-processing is complete.
            Default is False.
        **kwargs : dict
            Additional keyword arguments for future extensions or custom
            post-processing steps.
        """
        create_and_save_pae_plots(multimeric_object, output_dir)
        post_prediction_process(
            output_dir,
            zip_pickles=zip_pickles,
            remove_pickles=remove_pickles,
        )
