""" Implements structure prediction backend using AlphaLink2.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
            Valentin Maurer <valentin.maurer@embl-hamburg.de>
            
"""
import math, time
from absl import logging
from typing import Dict, List, Tuple, Union
import os
from os.path import join, exists, splitext
from shutil import copyfile
import re, json
from alphapulldown.folding_backend.alphafold_backend import _save_pae_json_file
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.plotting import plot_pae_from_matrix
from .folding_backend import FoldingBackend
import torch
import numpy as np
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.dataset import process_ap
from unifold.data import residue_constants, protein
from unifold.data.data_ops import get_pairwise_distances
from unicore.utils import (
    tensor_tree_map,
)
logging.set_verbosity(logging.INFO)

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
MODEL_NAME = 'multimer_af2_crop'
MAX_RECYCLING_ITERS = 3
NUM_ENSEMBLES = 1
SAMPLE_TEMPLATES = False
DATA_RANDOM_SEED = 42


class AlphaLinkBackend(FoldingBackend):
    """
    A backend class for running protein structure predictions using the AlphaLink model.
    """
    @staticmethod
    def setup(
        model_dir: str,
        model_name: str = "multimer_af2_crop",
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures an AlphaLink model runner including crosslinking data.

        Parameters
        ----------
        model_name : str
            The name of the model to use for prediction. Set to be multimer_af2_crop as used in AlphaLink2
        model_dir : str
            Path to either:
            1. A directory containing AlphaLink weights files (e.g., AlphaLink-Multimer_SDA_v3.pt)
            2. A specific AlphaLink weights file (e.g., /path/to/AlphaLink-Multimer_SDA_v3.pt)
            Expected file names: AlphaLink-Multimer_SDA_v2.pt or AlphaLink-Multimer_SDA_v3.pt
        crosslinks : str
            The path to the file containing crosslinking data.
        **kwargs : dict
            Additional keyword arguments for model configuration.

        Returns
        -------
        Dict
            A dictionary records the path to the AlphaLink2 neural network weights
            i.e. a pytorch checkpoint file, crosslink information,
            and Pytorch model configs

        Raises
        ------
        FileNotFoundError
            If the AlphaLink weights file does not exist at the specified path.
        ValueError
            If the file does not have a .pt extension or is not a recognized AlphaLink weights file.
        """
        # Check if model_dir is a file or directory
        if os.path.isfile(model_dir):
            # Direct file path provided
            weights_file = model_dir
        elif os.path.isdir(model_dir):
            # Directory provided, search for weights files
            expected_names = ["AlphaLink-Multimer_SDA_v3.pt", "AlphaLink-Multimer_SDA_v2.pt"]
            weights_file = None
            
            for filename in expected_names:
                candidate_path = os.path.join(model_dir, filename)
                if os.path.isfile(candidate_path):
                    weights_file = candidate_path
                    break
            
            if weights_file is None:
                # List all .pt files in directory for better error message
                pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                if pt_files:
                    raise FileNotFoundError(
                        f"AlphaLink2 weights file not found in directory: {model_dir}. "
                        f"Expected one of: {expected_names}. "
                        f"Found .pt files: {pt_files}"
                    )
                else:
                    raise FileNotFoundError(
                        f"AlphaLink2 weights file not found in directory: {model_dir}. "
                        f"Expected one of: {expected_names}. "
                        f"No .pt files found in directory."
                    )
        else:
            # Neither file nor directory exists
            raise FileNotFoundError(
                f"AlphaLink2 weights file or directory does not exist at: {model_dir}"
            )
        
        # Check if it's a .pt file
        if not weights_file.endswith(".pt"):
            raise ValueError(
                f"AlphaLink2 weights file must have .pt extension, got: {weights_file}"
            )
        
        # Check if it's a recognized AlphaLink weights file
        filename = os.path.basename(weights_file)
        expected_names = ["AlphaLink-Multimer_SDA_v2.pt", "AlphaLink-Multimer_SDA_v3.pt"]
        if filename not in expected_names:
            logging.warning(
                f"AlphaLink weights file name '{filename}' is not one of the expected names: {expected_names}. "
                f"This might still work if the file contains valid AlphaLink weights."
            )

        configs = model_config(model_name)

        return {
            "param_path": weights_file,
            "configs": configs,
        }

    @staticmethod
    def unload_tensors(batch, out):
        """
        A method to unload tensors from GPU to CPU
        Note: NO need to remove the recycling dimension as process_ap already removed it
        """
        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x
        batch = tensor_tree_map(to_float, batch)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
        return batch, out

    @staticmethod
    def prepare_model_runner(param_path: str, bf16: bool = False, model_device: str = '') -> AlphaFold:
        """
        A method that initialise AlphaFold PyTorch model

        Args:
        param_path : path to pretrained AlphaLink2 neural network weights
        bf16: use bf16 precision or not. Default is False
        model_device: device name

        Return:
        an AlphaFold object 
        """
        config = model_config(MODEL_NAME)
        config.data.common.max_recycling_iters = MAX_RECYCLING_ITERS
        config.globals.max_recycling_iters = MAX_RECYCLING_ITERS
        config.data.predict.num_ensembles = NUM_ENSEMBLES
        if SAMPLE_TEMPLATES:
            # enable template samples for diversity
            config.data.predict.subsample_templates = True
        model = AlphaFold(config)
        state_dict = torch.load(param_path)["ema"]["params"]
        state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(model_device)
        model.eval()
        model.inference_mode()
        if bf16:
            model.bfloat16()
        return model

    @staticmethod
    def check_resume_status(curr_model_name: str, output_dir: str) -> Tuple[bool, Union[None, float]]:
        """
        A method that check if the current model has been creatd already

        Args:
        curr_model_name : the name of the model to check, format: AlphaLink2_model_{it}_seed_{cur_seed}
        output_dir: output directory to save the results

        Return:
        already_exists: boolean indicating whether the model already exists
        iptm_value: if exists already, return its iptm values, otherwise, returns None
        """
        pattern = r'_(\d+\.\d+)\.pdb$'
        curr_pdb_file = [i for i in os.listdir(output_dir) if i.startswith(
            curr_model_name) and i.endswith(".pdb")]
        curr_pae_json = [i for i in os.listdir(output_dir) if i.startswith(
            f"pae_{curr_model_name}") and i.endswith(".json")]
        if len(curr_pdb_file) == 1 and len(curr_pae_json) == 1:
            matched = re.search(pattern, curr_pdb_file[0])
            iptm_value = float(matched.group(1))
            already_exists = True
        else:
            iptm_value = None
            already_exists = False
        return already_exists, iptm_value
    
    @staticmethod
    def preprocess_features(feature_dict):
        """
        Preprocess features to ensure compatibility with AlphaLink2.
        
        AlphaLink2 expects certain features to be scalars, but AlphaPulldown
        may provide them as arrays. This method converts arrays to scalars
        where needed and adds missing features.
        """
        processed_features = feature_dict.copy()
        
        # Convert seq_length from array to scalar if needed
        if "seq_length" in processed_features:
            seq_length = processed_features["seq_length"]
            try:
                if hasattr(seq_length, "__len__") and hasattr(seq_length, "__iter__") and len(seq_length) > 1:
                    # If seq_length is an array, take the first value
                    processed_features["seq_length"] = seq_length[0]
            except (TypeError, ValueError):
                # seq_length is a scalar, leave it as is
                pass
        
        # Convert other potential array features to scalars
        scalar_features = ["num_alignments", "num_templates"]
        for feature_name in scalar_features:
            if feature_name in processed_features:
                feature_value = processed_features[feature_name]
                try:
                    if hasattr(feature_value, "__len__") and hasattr(feature_value, "__iter__") and len(feature_value) > 1:
                        processed_features[feature_name] = feature_value[0]
                except (TypeError, ValueError):
                    # feature_value is a scalar, leave it as is
                    pass
        
        # Handle template feature key name differences
        if "template_all_atom_masks" in processed_features and "template_all_atom_mask" not in processed_features:
            processed_features["template_all_atom_mask"] = processed_features["template_all_atom_masks"]
        
        # Convert template features from one-hot to integer encoding if needed
        if "template_aatype" in processed_features:
            template_aatype = processed_features["template_aatype"]
            if len(template_aatype.shape) == 3:
                # Convert one-hot encoding to integer encoding
                processed_features["template_aatype"] = np.argmax(template_aatype, axis=-1)
        
        # Fix template_sum_probs shape if needed
        if "template_sum_probs" in processed_features:
            template_sum_probs = processed_features["template_sum_probs"]
            if len(template_sum_probs.shape) == 1:
                # Reshape to match expected schema: (1, 1) instead of (1,)
                processed_features["template_sum_probs"] = template_sum_probs.reshape(1, 1)
        
        # Add missing features that AlphaLink2 expects
        seq_len = processed_features.get("seq_length", 0)
        if isinstance(seq_len, (list, np.ndarray)):
            try:
                seq_len = seq_len[0] if len(seq_len) > 0 else 0
            except (TypeError, ValueError):
                # seq_len is a scalar, leave it as is
                pass
        
        # Add deletion_matrix if missing
        if "deletion_matrix" not in processed_features:
            if "deletion_matrix_int" in processed_features:
                processed_features["deletion_matrix"] = processed_features["deletion_matrix_int"]
            else:
                processed_features["deletion_matrix"] = np.zeros((1, seq_len))
        
        # Add extra_deletion_matrix if missing
        if "extra_deletion_matrix" not in processed_features:
            if "deletion_matrix_int_all_seq" in processed_features:
                processed_features["extra_deletion_matrix"] = processed_features["deletion_matrix_int_all_seq"]
            else:
                processed_features["extra_deletion_matrix"] = np.zeros((1, seq_len))
        
        # Add msa_mask if missing
        if "msa_mask" not in processed_features:
            if "msa" in processed_features:
                msa_shape = processed_features["msa"].shape
                processed_features["msa_mask"] = np.ones(msa_shape)
            else:
                processed_features["msa_mask"] = np.ones((1, seq_len))
        
        # Add msa_row_mask if missing
        if "msa_row_mask" not in processed_features:
            if "msa" in processed_features:
                msa_shape = processed_features["msa"].shape
                processed_features["msa_row_mask"] = np.ones((msa_shape[0],))
            else:
                processed_features["msa_row_mask"] = np.ones((1,))
        
        # Add multimer-specific features if missing
        if "asym_id" not in processed_features:
            processed_features["asym_id"] = np.zeros(seq_len, dtype=np.int32)
        
        if "entity_id" not in processed_features:
            processed_features["entity_id"] = np.zeros(seq_len, dtype=np.int32)
        
        if "sym_id" not in processed_features:
            processed_features["sym_id"] = np.ones(seq_len, dtype=np.int32)
        
        return processed_features

    @staticmethod
    def automatic_chunk_size(seq_len, device, is_bf16 = False):
        def get_device_mem(device):
            if device != "cpu" and torch.cuda.is_available():
                cur_device = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
                total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024
                return total_memory_in_GB
            else:
                return 40
        total_mem_in_GB = get_device_mem(device)
        factor = math.sqrt(total_mem_in_GB/40.0*(0.55 * is_bf16 + 0.45))*0.95
        if seq_len < int(1024*factor):
            chunk_size = 256
            block_size = None
        elif seq_len < int(2048*factor):
            chunk_size = 128
            block_size = None
        elif seq_len < int(3072*factor):
            chunk_size = 64
            block_size = None
        elif seq_len < int(4096*factor):
            chunk_size = 32
            block_size = 512
        else:
            chunk_size = 4
            block_size = 256
        return chunk_size, block_size

    @staticmethod
    def predict_iterations(feature_dict, output_dir='', param_path='', input_seqs=[],
                           configs=None, crosslinks='', chain_id_map=None,
                           num_inference=10, resume=True,
                           cutoff=25) -> None:
        """Modified from 
            
        https://github.com/Rappsilber-Laboratory/AlphaLink2/blob/main/inference.py
        """
        if torch.cuda.is_available():
            model_device = 'cuda:0'
        else:
            model_device = 'cpu'
        model = AlphaLinkBackend.prepare_model_runner(
            param_path, model_device=model_device)

        for it in range(num_inference):
            cur_seed = hash((DATA_RANDOM_SEED, it)) % 100000
            curr_model_name = f"AlphaLink2_model_{it}_seed_{cur_seed}"
            already_exists, iptm_value = AlphaLinkBackend.check_resume_status(
                curr_model_name, output_dir)
            if resume and already_exists:
                print(
                    f"{curr_model_name} already done with iptm: {iptm_value}. Skipped.")
                continue
            else:
                # Preprocess features to ensure compatibility with AlphaLink2
                processed_features = AlphaLinkBackend.preprocess_features(feature_dict)
                
                # Convert empty crosslinks string to None
                crosslinks_param = None if crosslinks == "" else crosslinks
                
                batch, _ = process_ap(config=configs.data,
                                      features=processed_features,
                                      mode="predict", labels=None,
                                      seed=cur_seed, batch_idx=None,
                                      data_idx=None, is_distillation=False,
                                      chain_id_map=chain_id_map,
                                      crosslinks=crosslinks_param)
                # faster prediction with large chunk/block size
                seq_len = batch["aatype"].shape[-1]
                chunk_size, block_size = AlphaLinkBackend.automatic_chunk_size(
                    seq_len,
                    model_device
                )
                model.globals.chunk_size = chunk_size
                model.globals.block_size = block_size

                with torch.no_grad():
                    batch = {
                        k: torch.as_tensor(v, device=model_device)
                        for k, v in batch.items()
                    }

                    t = time.perf_counter()
                    torch.autograd.set_detect_anomaly(True)
                    raw_out = model(batch)
                    print(f"Inference time: {time.perf_counter() - t}")
                    score = ["plddt", "ptm", "iptm",
                             "iptm+ptm", 'predicted_aligned_error']
                    out = {
                        k: v for k, v in raw_out.items()
                        if k.startswith("final_") or k in score
                    }
                    batch, out = AlphaLinkBackend.unload_tensors(batch, out)
                    ca_idx = residue_constants.atom_order["CA"]
                    ca_coords = torch.from_numpy(
                        out["final_atom_positions"][..., ca_idx, :])
                    distances = get_pairwise_distances(ca_coords)  # [0]#[0,0]
                    xl = torch.from_numpy(
                        batch['xl'][..., 0].astype(np.int32) > 0)
                    interface = torch.from_numpy(
                        batch['asym_id'][..., None] != batch['asym_id'][..., None, :])
                    satisfied = torch.sum(
                        distances[xl[0] & interface[0]] <= cutoff) / 2
                    total_xl = torch.sum(xl & interface) / 2
                    print("Current seed: %d Model %d Crosslink satisfaction: %.3f Model confidence: %.3f" % (
                        cur_seed, it, satisfied / total_xl, np.mean(out["iptm+ptm"])))
                    plddt = out["plddt"]
                    plddt_b_factors = np.repeat(
                        plddt[..., None], residue_constants.atom_type_num, axis=-1
                    )
                    # Weird bug in AlphaLink2 where asym_id is 9 instead of 'A' for monomers
                    print(f"{batch['asym_id']}")
                    if 'asym_id' in batch:
                        if batch['asym_id'].all() == '9':
                            batch['asym_id'] = 'A'
                    cur_protein = protein.from_prediction(
                        features=batch, result=out, b_factors=plddt_b_factors
                    )
                    iptm_value = np.mean(out["iptm+ptm"])
                    cur_save_name = (
                        f"AlphaLink2_model_{it}_seed_{cur_seed}_{iptm_value:.3f}.pdb"
                    )
                    cur_plot_name = f"AlphaLink2_model_{it}_seed_{cur_seed}_{iptm_value:.3f}_pae.png"
                    
                    # Ensure output directory exists before saving files
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # save pae json file
                    _save_pae_json_file(out['predicted_aligned_error'], str(np.max(out['predicted_aligned_error'])),
                                        output_dir, f"AlphaLink2_model_{it}_seed_{cur_seed}_{iptm_value:.3f}")
                    # plot PAE
                    plot_pae_from_matrix(input_seqs,
                                         pae_matrix=out['predicted_aligned_error'],
                                         figure_name=join(output_dir, cur_plot_name))
                    cur_protein.chain_index = np.squeeze(
                        cur_protein.chain_index, 0)
                    cur_protein.aatype = np.squeeze(cur_protein.aatype, 0)
                    unique_asym_ids = np.unique(cur_protein.chain_index)
                    seq_lens = [np.sum(cur_protein.chain_index == u)
                                for u in unique_asym_ids]
                    residue_index = []
                    for seq_len in seq_lens:
                        residue_index += range(seq_len)
                    cur_protein.residue_index = np.array(residue_index)
                    with open(join(output_dir, cur_save_name), "w") as f:
                        f.write(protein.to_pdb(cur_protein))

                    del out

    @staticmethod
    def predict(
        objects_to_model: List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, str]]],
        **kwargs,
    ):
        """
        Predicts the structure of proteins using configured AlphaLink models.

        Parameters
        ----------
        objects_to_model : List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, str]]]
            A list of dictionaries. Each dictionary has keys 'object' and 'output_dir'.
            The 'object' key contains an instance of MultimericObject, MonomericObject, or ChoppedObject.
            The 'output_dir' key contains the corresponding output directory to save the modelling results.
        **kwargs : dict
            Additional keyword arguments including:
            - configs: Configuration dictionary for the AlphaLink model
            - param_path: Path to the AlphaLink model parameters
            - crosslinks: Path to crosslink information pickle for AlphaLink
        """
        # Extract required parameters from kwargs
        configs = kwargs.get('configs')
        param_path = kwargs.get('param_path')
        crosslinks = kwargs.get('crosslinks')
        
        if not all([configs, param_path]):
            raise ValueError("Missing required parameters: configs or param_path")
        
        # Make crosslinks optional - if not provided, use empty string
        if crosslinks is None:
            crosslinks = ""
        
        logging.warning(f"You chose to model with AlphaLink2 via AlphaPulldown. Please also cite:K.Stahl,O.Brock and J.Rappsilber, Modelling protein complexes with crosslinking mass spectrometry and deep learning, 2023, doi: 10.1101/2023.06.07.544059")
        
        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # Get num_predictions_per_model from kwargs, default to 1
            num_predictions_per_model = kwargs.get('num_predictions_per_model', 1)
            
            # Get chain_id_map if available, otherwise create a default one
            chain_id_map = getattr(object_to_model, 'chain_id_map', None)
            if chain_id_map is None:
                # Create a default chain_id_map for single chain proteins
                # AlphaLink2 expects objects with descriptions, not just integers
                from dataclasses import dataclass
                
                @dataclass
                class ChainInfo:
                    description: str
                    sequence: str
                
                # Create a simple chain_id_map with proper objects
                if hasattr(object_to_model, 'input_seqs') and object_to_model.input_seqs:
                    # Use the first sequence as default
                    sequence = object_to_model.input_seqs[0] if object_to_model.input_seqs else ""
                    description = "default_chain"
                else:
                    sequence = ""
                    description = "default_chain"
                
                chain_id_map = {'A': ChainInfo(description=description, sequence=sequence)}
            else:
                # If chain_id_map exists but contains integers, convert them to proper objects
                if chain_id_map and isinstance(next(iter(chain_id_map.values())), int):
                    from dataclasses import dataclass
                    
                    @dataclass
                    class ChainInfo:
                        description: str
                        sequence: str
                    
                    # Convert integer-based chain_id_map to object-based
                    converted_chain_id_map = {}
                    for chain_id, chain_idx in chain_id_map.items():
                        if hasattr(object_to_model, 'input_seqs') and object_to_model.input_seqs:
                            sequence = object_to_model.input_seqs[chain_idx] if chain_idx < len(object_to_model.input_seqs) else ""
                        else:
                            sequence = ""
                        description = f"chain_{chain_id}"
                        converted_chain_id_map[chain_id] = ChainInfo(description=description, sequence=sequence)
                    chain_id_map = converted_chain_id_map
            
            AlphaLinkBackend.predict_iterations(object_to_model.feature_dict,output_dir,
                                                configs=configs,crosslinks=crosslinks,
                                                input_seqs=object_to_model.input_seqs,
                                                chain_id_map=chain_id_map,
                                                param_path=param_path,
                                                num_inference=num_predictions_per_model,
            )
            yield {'object': object_to_model, 
                   'prediction_results': "",
                   'output_dir': output_dir}

    @staticmethod
    def postprocess(prediction_results: Dict,
                    output_dir: str,
                    **kwargs: Dict) -> None:
        print(f"DEBUG: AlphaLink postprocess called with output_dir: {output_dir}")
        """
        Post-prediction process that makes AlphaLink2 results within 
        a sub-directory compatible with the analysis_pipeline

        Args:
            prediction_results: A dictionary from predict()
            output_dir: current corresponding output directory
        """
        def obtain_model_names_and_scores(pdb_file:str):
            pattern = r'AlphaLink2_model_(\d+)_seed_(\d+)_(\d+\.\d+)\.pdb$'
            matched = re.search(pattern, pdb_file)
            iptm_value = matched.group(3)
            # Extract just the filename without path
            model_name = splitext(os.path.basename(pdb_file))[0]
            return (model_name, float(iptm_value))    
        
        def make_ranked_pdb_files(outputdir: str, order: list):
            for idx, model_name in enumerate(order):
                new_file_name = join(outputdir, f"ranked_{idx}.pdb")
                # Find the actual file path in subdirectories
                old_file_name = None
                for root, dirs, files in os.walk(outputdir):
                    for file in files:
                        if file == f"{model_name}.pdb":
                            old_file_name = join(root, file)
                            break
                    if old_file_name:
                        break
                
                if old_file_name and exists(old_file_name):
                    copyfile(old_file_name, new_file_name)

        def create_ranking_debug_json(model_and_qualities:dict) -> Tuple[tuple, list]:
            """A function to create ranking_debug.json based on the iptm-ptm score"""
            sorted_dict = sorted(model_and_qualities.items(), key=lambda x: x[1], reverse=True)
            order = [i[0] for i in sorted_dict]
            iptm_ptm = [i[1] for i in sorted_dict]
            return json.dumps({"iptm+ptm": iptm_ptm, "order":order}), order
            
        model_and_qualities = dict()
        
        # Look for PDB files in the output directory and its subdirectories
        all_pdb_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.startswith("AlphaLink2_model_") and file.endswith(".pdb"):
                    all_pdb_files.append(os.path.join(root, file))
        
        model_and_qualities.update({model_and_values[0]: model_and_values[1] for model_and_values in [obtain_model_names_and_scores(i) for i in all_pdb_files]})
        ranking_debug_json,order = create_ranking_debug_json(model_and_qualities)
        make_ranked_pdb_files(output_dir, order)
        
        # Write ranking_debug.json to the main output directory
        with open(join(output_dir, "ranking_debug.json"),"w") as outfile:
            outfile.write(ranking_debug_json)
            outfile.close()