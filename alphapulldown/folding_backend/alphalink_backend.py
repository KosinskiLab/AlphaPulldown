""" Implements structure prediction backend using AlphaLink2.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
            Valentin Maurer <valentin.maurer@embl-hamburg.de>
            
"""
import math, time
from absl import logging
from typing import Dict, List, Tuple, Union
from os import listdir, makedirs
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
            Path to the pytorch checkpoint that corresponds to the neural network weights from AlphaLink2.
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
        """
        if not exists(model_dir):
            raise FileNotFoundError(
                f"AlphaLink2 network weight does not exist at: {model_dir}"
            )
        if not model_dir.endswith(".pt"):
            f"{model_dir} does not seem to be a pytorch checkpoint."

        configs = model_config(model_name)

        return {
            "param_path": model_dir,
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
        curr_pdb_file = [i for i in listdir(output_dir) if i.startswith(
            curr_model_name) and i.endswith(".pdb")]
        curr_pae_json = [i for i in listdir(output_dir) if i.startswith(
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
                batch, _ = process_ap(config=configs.data,
                                      features=feature_dict,
                                      mode="predict", labels=None,
                                      seed=cur_seed, batch_idx=None,
                                      data_idx=None, is_distillation=False,
                                      chain_id_map=chain_id_map,
                                      crosslinks=crosslinks)
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
                    cur_protein = protein.from_prediction(
                        features=batch, result=out, b_factors=plddt_b_factors
                    )
                    iptm_value = np.mean(out["iptm+ptm"])
                    cur_save_name = (
                        f"AlphaLink2_model_{it}_seed_{cur_seed}_{iptm_value:.3f}.pdb"
                    )
                    cur_plot_name = f"AlphaLink2_model_{it}_seed_{cur_seed}_{iptm_value:.3f}_pae.png"
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
        configs: Dict,
        param_path: str,
        crosslinks: str,
        objects_to_model: List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, str]]],
        **kwargs,
    ):
        """
        Predicts the structure of proteins using configured AlphaLink models.

        Parameters
        ----------
        configs : Dict
            Configuration dictionary for the AlphaLink model obtained from
            py:meth:`AlphaLinkBackend.setup`.
        param_path : str
            Path to the AlphaLink model parameters.
        crosslinks : str
            Path to crosslink information pickle for AlphaLink.
        objects_to_model : List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, str]]]
            A list of dictionaries. Each dictionary has keys 'object' and 'output_dir'.
            The 'object' key contains an instance of MultimericObject, MonomericObject, or ChoppedObject.
            The 'output_dir' key contains the corresponding output directory to save the modelling results.
        **kwargs : dict
            Additional keyword arguments for prediction.
        """
        logging.warning(f"You chose to model with AlphaLink2 via AlphaPulldown. Please also cite:K.Stahl,O.Brock and J.Rappsilber, Modelling protein complexes with crosslinking mass spectrometry and deep learning, 2023, doi: 10.1101/2023.06.07.544059")
        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']
            makedirs(output_dir, exist_ok=True)
            AlphaLinkBackend.predict_iterations(object_to_model.feature_dict,output_dir,
                                                configs=configs,crosslinks=crosslinks,
                                                input_seqs=object_to_model.input_seqs,
                                                chain_id_map=object_to_model.chain_id_map,
                                                param_path=param_path,
            )
            yield {'object': object_to_model, 
                   'prediction_results': "",
                   'output_dir': output_dir}

    @staticmethod
    def postprocess(prediction_results: Dict,
                    output_dir: str,
                    **kwargs: Dict) -> None:
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
            model_name = splitext(pdb_file)[0]
            return (model_name, float(iptm_value))    
        
        def make_ranked_pdb_files(outputdir: str, order: list):
            for idx, model_name in enumerate(order):
                new_file_name = join(outputdir, f"ranked_{idx}.pdb")
                old_file_name = join(outputdir, f"{model_name}.pdb")
                copyfile(old_file_name, new_file_name)

        def create_ranking_debug_json(model_and_qualities:dict) -> Tuple[tuple, list]:
            """A function to create ranking_debug.json based on the iptm-ptm score"""
            sorted_dict = sorted(model_and_qualities.items(), key=lambda x: x[1], reverse=True)
            order = [i[0] for i in sorted_dict]
            iptm_ptm = [i[1] for i in sorted_dict]
            return json.dumps({"iptm+ptm": iptm_ptm, "order":order}), order
            
        model_and_qualities = dict()
        all_pdb_files = [i for i in listdir(output_dir) if i.startswith("AlphaLink2_model_") and i.endswith(".pdb")]
        model_and_qualities.update({model_and_values[0]: model_and_values[1] for model_and_values in [obtain_model_names_and_scores(i) for i in all_pdb_files]})
        ranking_debug_json,order = create_ranking_debug_json(model_and_qualities)
        make_ranked_pdb_files(output_dir, order)
        with open(join(output_dir, "ranking_debug.json"),"w") as outfile:
            outfile.write(ranking_debug_json)
            outfile.close()