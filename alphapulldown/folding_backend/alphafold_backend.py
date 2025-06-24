""" Implements structure prediction backend using AlphaFold.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
            Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de> 
            Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
import time
import json
import pickle
import subprocess
import enum
from typing import Dict, Union, List, Any
import os
from absl import logging
import numpy as np
from copy import copy
import jax.numpy as jnp
from alphapulldown.utils.plotting import plot_pae_from_matrix
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.post_modelling import post_prediction_process
#from alphapulldown.utils.calculate_rmsd import calculate_rmsd_and_superpose
from alphapulldown.utils.modelling_setup import pad_input_features
from alphafold.relax import relax
from alphafold.common import protein, residue_constants, confidence
from .folding_backend import FoldingBackend
logging.set_verbosity(logging.INFO)


# Relaxation parameters.
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2


def _jnp_to_np(output):
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def _save_pae_json_file(pae: np.ndarray, max_pae: float, output_dir: str, model_name: str) -> None:
    """
    Check prediction result for PAE data and save to a JSON file if present.

    Args:
    pae: The n_res x n_res PAE array.
    max_pae: The maximum possible PAE value.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
    """
    pae_json = confidence.pae_json(pae, max_pae)
    pae_json_output_path = os.path.join(output_dir, f'pae_{model_name}.json')
    with open(pae_json_output_path, 'w') as f:
        f.write(pae_json)


def _save_confidence_json_file(plddt: np.ndarray, output_dir: str, model_name: str) -> None:
    """
    Check prediction result for confidence data and save to a JSON file if present.
    Args:
        plddt: The n_res x 1 pLDDT array.
        output_dir: Directory to which files are saved.
        model_name: Name of a model.
    """
    confidence_json = confidence.confidence_json(plddt)
    confidence_json_output_path = os.path.join(
        output_dir, f'confidence_{model_name}.json')
    with open(confidence_json_output_path, 'w') as f:
        f.write(confidence_json)


def _read_from_json_if_exists(json_path: str) -> Dict:
    """Reads a JSON file or creates it with default data if it does not exist."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data


def _reset_template_features(feature_dict: Dict) -> None:
    """
    Resets specific features within a dictionary to their default state.
    - 'template_aatype' and 'template_all_atom_positions' are reset to zeros.
    - 'template_all_atom_masks' is reset to ones.
    - 'num_templates' is set to one.
    Parameters:
    feature_dict (Dict[str, np.ndarray]): The feature dictionary to be modified.
    """
    seq_length = feature_dict["seq_length"]
    for key, value in feature_dict.items():
        if key == "template_aatype":
            feature_dict[key] = np.zeros((1, seq_length), dtype='int64')
        elif key == "template_all_atom_positions":
            feature_dict[key] = np.zeros(
                (1, seq_length, 37, 3), dtype='float32')
        elif key == "template_all_atom_mask":
            feature_dict[key] = np.ones((1, seq_length, 37), dtype='float32')
        elif key == "num_templates":
            feature_dict[key] = np.ones_like(value)


class AlphaFoldBackend(FoldingBackend):
    """
    A backend to perform structure prediction using AlphaFold.
    """

    @staticmethod
    def setup(
        model_name: str,
        num_cycle: int,
        model_dir: str,
        num_predictions_per_model: int,
        msa_depth_scan=False,
        model_names_custom: List[str] = None,
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
        num_predictions_per_model : int
            The number of multimer predictions to perform for each model.
        msa_depth_scan : bool, optional
            Whether to adjust MSA depth logarithmically, default is False.
        model_names_custom : list, optional
            A list of strings that specify which models to run, default is None, meaning all 5 models will be used
        msa_depth : int or None, optional
            A specific MSA depth to use, default is None.
        allow_resume : bool, optional
            If set to True, resumes prediction from partially completed runs, default is True.
        **kwargs : dict
            Additional keyword arguments for model runner configuration.

        Returns
        -------
        Dict
            A dictionary containing the configured model runners, and other settings

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
        # add model names of older versionsto be compatible with older version of AlphaFold Multimer
        old_model_names = (
          'model_1_multimer',
          'model_2_multimer',
          'model_3_multimer',
          'model_4_multimer',
          'model_5_multimer',
          'model_1_multimer_v2',
          'model_2_multimer_v2',
          'model_3_multimer_v2',
          'model_4_multimer_v2',
          'model_5_multimer_v2',
      )
        if model_names_custom:
            model_names_custom = tuple(model_names_custom)
            if all(x in model_names + old_model_names for x in model_names_custom):
                model_names = model_names_custom
            else:
                raise Exception(
                    f"Provided model names {model_names_custom} not part of available {model_names + old_model_names}"
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
                        num_predictions_per_model,
                    )
                ).astype(int)

                extra_msa_ranges = np.rint(
                    np.logspace(
                        np.log10(32),
                        np.log10(num_extra_msa),
                        num_predictions_per_model,
                    )
                ).astype(int)

            for i in range(num_predictions_per_model):
                logging.debug(f"msa_depth is type : {type(msa_depth)} value: {msa_depth}")
                logging.debug(f"msa_depth_scan is type: {type(msa_depth_scan)} value: {msa_depth_scan}")
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

        return {"model_runners": model_runners}

    @staticmethod
    def predict_individual_job(
        model_runners: Dict,
        multimeric_object: Union[MultimericObject, MonomericObject],
        allow_resume: bool,
        skip_templates: bool,
        output_dir: Dict,
        random_seed: int,
        **kwargs,
    ) -> dict:
        """
        Executes structure predictions using configured AlphaFold models on one individual job.

        Parameters
        ----------
        model_runners : Dict
            Configured model runners with model names as keys.
        allow_resume : bool
            If set to True, resumes prediction from partially completed runs.
        skip_templates : bool
            Do not use templates for prediction.
        multimeric_object : MultimericObject
            An object containing features of the multimeric proteins or monomeric protein,
            for the sake of simplicity, it is named as multimeric_object but can be a MonomericObject.
        output_dir : str
            The directory to save prediction results and PDB files.
        random_seed : int, optional
            A seed for random number generation to ensure reproducibility, default is 42.
        **kwargs : dict
            Additional keyword arguments for prediction.

        Returns
        -------
        Dict
            A dictionary mapping model names with corresponding prediction results and predicted proteins.

        Raises
        ------
        ValueError
            If multimeric mode is enabled but no valid templates are found.
        ValueError
            If multimeric mode and skip templates are enabled at the same time
        """
        timings = {}
        prediction_results = {}
        START = 0
        multimeric_mode = multimeric_object.multimeric_mode if hasattr(multimeric_object, "multimeric_mode") else None
        original_feature_dict = copy(multimeric_object.feature_dict)
        if allow_resume:
            logging.info(
            f"Now running predictions on {multimeric_object.description}. Checking existing results...")
            for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
                unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
                result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
                # If --remove_result_pickle=True the predictions are done anew.
                if os.path.exists(unrelaxed_pdb_path) and os.path.exists(result_output_path):
                    result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
                    with open(result_output_path, "rb") as f:
                        prediction_result = pickle.load(f)
                        # Update prediction_result with input seqs and unrelaxed protein
                        if "seqs" not in prediction_result.keys():
                            prediction_result.update(
                                {"seqs": multimeric_object.input_seqs if hasattr(multimeric_object, "input_seqs") else [
                                    multimeric_object.sequence]})
                        if "unrelaxed_protein" not in prediction_result.keys():
                            plddt_b_factors = np.repeat(
                                prediction_result['plddt'][:,
                                None], residue_constants.atom_type_num, axis=-1
                            )
                            model_random_seed = model_index + random_seed * len(model_runners)
                            logging.info(f"Running prediction using model {model_name} "
                                         f"and random seed: {model_random_seed}")
                            processed_feature_dict = model_runner.process_features(
                                original_feature_dict, random_seed=model_random_seed
                            )
                            unrelaxed_protein = protein.from_prediction(
                                features=processed_feature_dict,
                                result=prediction_result,
                                b_factors=plddt_b_factors,
                                remove_leading_feature_dimension=not model_runner.multimer_mode,
                            )
                            prediction_result.update(
                                {"unrelaxed_protein": unrelaxed_protein})
                    prediction_results.update({model_name: prediction_result})
                    START = model_index + 1
                else:
                    break
        logging.info(f"{START} models are already completed.")
        if START == len(model_runners):
            logging.info(
                f"All predictions for {multimeric_object.description} are already completed.")
            return prediction_results
        # first check whether the desired num_res and num_msa are specified for padding
        desired_num_res, desired_num_msa = kwargs.get(
            "desired_num_res", None), kwargs.get("desired_num_msa", None)
        if ((desired_num_res is not None) and (desired_num_msa is not None) and
                type(multimeric_object) == MultimericObject):
            # This means padding is required to speed up the process
            pad_input_features(feature_dict=original_feature_dict,
                               desired_num_msa=desired_num_msa, desired_num_res=desired_num_res)
            original_feature_dict['num_alignments'] = np.array([desired_num_msa])
        num_models = len(model_runners)
        # Predict
        for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
            t_0 = time.time()
            model_random_seed = model_index + random_seed * num_models
            processed_feature_dict = model_runner.process_features(
                original_feature_dict, random_seed=model_random_seed
            )
            # Skip if results.pkl und unrelaxed.pdb are exist
            if model_index < START: continue
            # TODO: re-predict models if --allow_resume and previous predictions were done with templates
            if skip_templates:
                _reset_template_features(processed_feature_dict)
            timings[f"process_features_{model_name}"] = time.time() - t_0
            # Die if --multimeric_mode=True but no non-zero templates are in the feature dict
            if multimeric_mode:
                if "template_all_atom_positions" in processed_feature_dict:
                    if not np.any(
                        processed_feature_dict["template_all_atom_positions"]
                    ):
                        if skip_templates:
                            raise ValueError(
                                "You cannot skip templates in multimeric mode.")
                        raise ValueError(
                            "No valid templates found: all positions are zero."
                        )
                else:
                    raise ValueError(
                        "No template_all_atom_positions key found in processed_feature_dict."
                    )
            t_0 = time.time()
            logging.info(
                f"Now running predictions on {multimeric_object.description} using {model_name}")
            prediction_result = model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed
            )
            t_diff = time.time() - t_0
            timings[f"predict_and_compile_{model_name}"] = t_diff
            logging.info(f"prediction costs : {t_diff} s")

            # Update prediction_result with input seqs and unrelaxed protein
            plddt_b_factors = np.repeat(
                prediction_result['plddt'][:,
                                           None], residue_constants.atom_type_num, axis=-1
            )
            unrelaxed_protein = protein.from_prediction(
                features=processed_feature_dict,
                result=prediction_result,
                b_factors=plddt_b_factors,
                remove_leading_feature_dimension=not model_runner.multimer_mode,
            )
            # Remove jax dependency from results
            np_prediction_result = _jnp_to_np(dict(prediction_result))
            # Save prediction results to pickle file
            result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
            with open(result_output_path, "wb") as f:
                 pickle.dump(np_prediction_result, f, protocol=4)
            prediction_result.update(
                        {"seqs": multimeric_object.input_seqs if hasattr(multimeric_object,"input_seqs") else [multimeric_object.sequence]})
            prediction_result.update({"unrelaxed_protein": unrelaxed_protein})
            prediction_results.update({model_name: prediction_result})
            # Save predictions to pdb files
            unrelaxed_pdb_path = os.path.join(
                output_dir, f"unrelaxed_{model_name}.pdb")

            with open(unrelaxed_pdb_path, "w") as f:
                f.write(protein.to_pdb(unrelaxed_protein))
            # Save timings to json file
            timings_output_path = os.path.join(output_dir, "timings.json")
            with open(timings_output_path, "w") as f:
                f.write(json.dumps(timings, indent=4))
        return prediction_results

    @staticmethod
    def predict(model_runners: Dict,
                objects_to_model: list,  # List of dicts with 'object' and 'output_dir'
                allow_resume: bool,
                skip_templates: bool,
                random_seed: int = 42,
                **kwargs):
        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']
            
            prediction_results = AlphaFoldBackend.predict_individual_job(
                model_runners=model_runners,
                multimeric_object=object_to_model,
                allow_resume=allow_resume,
                skip_templates=skip_templates,
                output_dir=output_dir,
                random_seed=random_seed,
                **kwargs
            )
            
            yield {'object': object_to_model, 
                   'prediction_results': prediction_results,
                   'output_dir': output_dir}
            
    @staticmethod
    def recalculate_confidence(prediction_results: Dict, multimer_mode:bool, 
                               total_num_res: int) -> Dict[str, Any]:
        """
        A method that remove pae values of padded residues and recalculate iptm_ptm score again 
        Modified based on https://github.com/KosinskiLab/alphafold/blob/c844e1bb60a3beb50bb8d562c6be046da1e43e3d/alphafold/model/model.py#L31
        """
        if type(prediction_results['predicted_aligned_error']) == np.ndarray:
            return prediction_results
        else:
            output = {}
            plddt = prediction_results['plddt'][:total_num_res]
            if 'predicted_aligned_error' in prediction_results:
                ptm = confidence.predicted_tm_score(
                logits=prediction_results['predicted_aligned_error']['logits'][:total_num_res,:total_num_res],
                breaks=prediction_results['predicted_aligned_error']['breaks'],
                asym_id=None)
                output['ptm'] = ptm

                pae = confidence.compute_predicted_aligned_error(
                logits=prediction_results['predicted_aligned_error']['logits'],
                breaks=prediction_results['predicted_aligned_error']['breaks'])
                max_pae = pae.pop('max_predicted_aligned_error')
                
                for k,v in pae.items():
                    output.update({k:v[:total_num_res, :total_num_res]})
                output['max_predicted_aligned_error'] = max_pae
                if multimer_mode:
                # Compute the ipTM only for the multimer model.
                    iptm = confidence.predicted_tm_score(
                    logits=prediction_results['predicted_aligned_error']['logits'][:total_num_res,:total_num_res],
                    breaks=prediction_results['predicted_aligned_error']['breaks'],
                    asym_id=prediction_results['predicted_aligned_error']['asym_id'][:total_num_res],
                    interface=True)
                    output.update({'iptm' : iptm})
                    ranking_confidence = 0.8 * iptm + 0.2 * ptm
                    output.update({'ranking_confidence' : ranking_confidence})
                if not multimer_mode:
                    # Monomer models use mean pLDDT for model ranking.
                    ranking_confidence =  np.mean(
                        plddt)
                    output.update({'ranking_confidence' : ranking_confidence})
                
                return output

    @staticmethod
    def postprocess(
        prediction_results: Dict,
        multimeric_object: MultimericObject,
        output_dir: str,
        features_directory: str,
        models_to_relax: ModelsToRelax,
        compress_pickles: bool = False,
        remove_pickles: bool = False,
        remove_keys_from_pickles: bool = False,
        convert_to_modelcif: bool = True,
        use_gpu_relax: bool = True,
        pae_plot_style: str = "red_blue",

        **kwargs: Dict,
    ) -> None:
        """
        Performs post-processing operations on predicted protein structures and
        writes results and plots to output_dir.

        Parameters
        ----------
        prediction_results: Dict
            A dictionary mapping model names with corresponding prediction results.
        multimeric_object : MultimericObject
            The multimeric object containing the predicted structures and
            associated data.
        output_dir : str
            The directory where post-processed files and plots will be saved.
        features_directory: str
            The directory containing the features used for prediction. Used by the convert_to_modelcif script.
        models_to_relax : object
            Specifies which models' predictions to relax, defaults to ModelsToRelax enum.
        compress_pickles : bool, optional
            If True, zips the pickle files containing prediction results.
            Default is False.
        remove_pickles : bool, optional
            If True, removes the pickle files after post-processing is complete.
            Default is False.
        convert_to_modelcif : bool, optional
            If set to True, converts all predicted models to ModelCIF format, default is True.
        use_gpu_relax : bool, optional
            If set to True, utilizes GPU acceleration for the relaxation step, default is True.
        pae_plot_style : str, optional
            The style of the PAE plot, red and blue or AF database style, default is "red_blue".
        **kwargs : dict
            Additional keyword arguments for future extensions or custom
            post-processing steps.
        """
        relaxed_pdbs = {}
        ranking_confidences = {}
        iptm_scores = {}
        ptm_scores = {}
        multimer_mode = type(multimeric_object) == MultimericObject
        # Read timings.json if exists
        timings_path = os.path.join(output_dir, 'timings.json')
        timings = _read_from_json_if_exists(timings_path)
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        relax_metrics = _read_from_json_if_exists(relax_metrics_path)

        ranking_path = os.path.join(output_dir, "ranking_debug.json")
        multimeric_mode = multimeric_object.multimeric_mode if hasattr(multimeric_object, "multimeric_mode") else None

        total_num_res = sum([len(s) for s in multimeric_object.input_seqs]) if multimer_mode else len(multimeric_object.sequence)
        # Save plddt json files.
        for model_name, prediction_result in prediction_results.items():
            prediction_result.update(AlphaFoldBackend.recalculate_confidence(prediction_result,multimer_mode,
                                                                         total_num_res))
            if 'unrelaxed_protein' in prediction_result.keys():
                unrelaxed_protein = prediction_result.pop("unrelaxed_protein")
            # Remove jax dependency from results
            np_prediction_result = _jnp_to_np(dict(prediction_result))
            # Save prediction results to pickle file
            result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
            with open(result_output_path, "wb") as f:
                pickle.dump(np_prediction_result, f, protocol=4)
            prediction_results[model_name]['unrelaxed_protein'] = unrelaxed_protein
            if 'iptm' in prediction_result:
                label = 'iptm+ptm'
                iptm_scores[model_name] = float(prediction_result['iptm'])
                cmplx = True
            else:
                label = 'plddts'
                ptm_scores[model_name] = float(prediction_result['ptm'])
                cmplx = False
            plddt = prediction_result['plddt']
            _save_confidence_json_file(plddt, output_dir, model_name)
            ranking_confidences[model_name] = prediction_result['ranking_confidence']
            # Save and plot PAE if predicting multimer.
            if (
                    'predicted_aligned_error' in prediction_result
                    and 'max_predicted_aligned_error' in prediction_result
            ):
                pae = prediction_result['predicted_aligned_error']

                max_pae = prediction_result['max_predicted_aligned_error']
                _save_pae_json_file(pae, float(max_pae),
                                    output_dir, model_name)

        # Rank by model confidence.
        ranked_order = [
            model_name for model_name, confidence in
            sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

        # Save pae plots as *.png files.
        for idx, model_name in enumerate(ranked_order):
            prediction_result = prediction_results[model_name]
            figure_name = os.path.join(
                output_dir, f"pae_plot_ranked_{idx}.png")
            plot_pae_from_matrix(
                seqs=prediction_result['seqs'],
                pae_matrix=prediction_result['predicted_aligned_error'],
                figure_name=figure_name,
                ranking=idx
            )

        # Save ranking_debug.json.
        with open(ranking_path, 'w') as f:
            json.dump(
                {label: ranking_confidences, 'order': ranked_order,
                 "iptm" if cmplx else "ptm": iptm_scores if cmplx else ptm_scores},
                f,
                indent=4
            )

        # Relax.
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
            if f'relax_{model_name}' in timings:
                continue
            t_0 = time.time()
            if 'unrelaxed_protein' in prediction_results[model_name].keys():
                unrelaxed_protein = prediction_results[model_name]['unrelaxed_protein']
            else:
                unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
                if not os.path.exists(unrelaxed_pdb_path):
                    logging.error(f"Cannot find {unrelaxed_pdb_path} for relaxation! Skipping...")
                    continue
                unrelaxed_pdb_string = open(unrelaxed_pdb_path, 'r').read()
                unrelaxed_protein = protein.from_pdb_string(unrelaxed_pdb_string)
            relaxed_pdb_str, _, violations = amber_relaxer.process(
                prot=unrelaxed_protein)
            relax_metrics[model_name] = {
                'remaining_violations': violations,
                'remaining_violations_count': sum(violations)
            }
            timings[f'relax_{model_name}'] = time.time() - t_0
            relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
            with open(relax_metrics_path, 'w') as f:
                f.write(json.dumps(relax_metrics, indent=4))
            relaxed_pdbs[model_name] = relaxed_pdb_str
            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(
                output_dir, f'relaxed_{model_name}.pdb')
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)

        with open(timings_path, 'w') as f:
            f.write(json.dumps(timings, indent=4))
              # Write out PDBs in rank order.
        for idx, model_name in enumerate(ranked_order):
            if model_name in relaxed_pdbs:
                protein_instance = relaxed_pdbs[model_name]
            else:
                protein_instance = protein.to_pdb(
                    prediction_results[model_name]['unrelaxed_protein'])
            ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
            with open(ranked_output_path, 'w') as f:
                f.write(protein_instance)
        # Extract multimeric template if multimeric mode is enabled.
        # if multimeric_mode:
        #     feature_dict = multimeric_object.feature_dict
        #     template_mask = feature_dict["template_all_atom_mask"][0]
        #     template_protein = protein.Protein(
        #         atom_positions=feature_dict["template_all_atom_positions"][0],
        #         atom_mask=template_mask,
        #         aatype=feature_dict["template_aatype"][0],
        #         residue_index=feature_dict.get("residue_index", None),
        #         chain_index=feature_dict["asym_id"],
        #         b_factors=np.zeros(template_mask.shape, dtype=float),
        #     )
        #     logging.info(f"template_aatype: {feature_dict['template_aatype'][0]}")
        #     pdb_string = protein.to_pdb(template_protein)
        #     # Check RMSD between the predicted model and the multimeric template.
        #     with tempfile.TemporaryDirectory() as temp_dir:
        #         template_file_path = f"{temp_dir}/template.pdb"
        #         with open(template_file_path, "w") as file:
        #             file.write(pdb_string)
        #         # TODO: use template_sequence for alignment
        #         calculate_rmsd_and_superpose(
        #             template_file_path, ranked_output_path, temp_dir
        #         )

        # Call convert_to_modelcif script
        if convert_to_modelcif:
            parent_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
            logging.info(f"Converting {output_dir} to ModelCIF format...")
            command = f"python3 {parent_dir}/scripts/convert_to_modelcif.py " \
                      f"--ap_output {output_dir} "

            result = subprocess.run(command,
                                   check=False,
                                   shell=True,
                                   capture_output=True,
                                   text=True)

            if result.stderr:
                logging.error(f"Error: {result.stderr}")
            else:
                logging.info("All PDBs converted to ModelCIF format.")
        post_prediction_process(
           output_dir,
           compress_pickles=compress_pickles,
           remove_pickles=remove_pickles,
           remove_keys=remove_keys_from_pickles
        )
