import gzip
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import pickle
from absl import logging
import os
import json
from typing import List, Tuple
import numpy as np


def obtain_seq_lengths(result_subdir: str) -> List[int]:
    """A function that obtain length of each chain in ranked_0.pdb"""
    best_result_pdb = os.path.join(result_subdir, "ranked_0.pdb")
    seq_length = []
    pdb_parser = PDBParser()
    sequence_builder = PPBuilder()
    if not os.path.exists(best_result_pdb):
        raise FileNotFoundError(
            f"ranked_0.pdb is not found at {result_subdir}")
    else:
        structure = pdb_parser.get_structure("ranked_0", best_result_pdb)
        seqs = sequence_builder.build_peptides(structure)
        for seq in seqs:
            seq_length.append(len(seq.get_sequence()))
    return seq_length

def obtain_pae_and_iptm(result_subdir: str, best_model: str) -> Tuple[np.array, float]:
    """A function that obtains PAE matrix and iptm score from either results pickle, zipped pickle, or pae_json"""
    try:
        ranking_results = json.load(
            open(os.path.join(result_subdir, "ranking_debug.json")))
    except FileNotFoundError as e:
        logging.warning(f"ranking_debug.json is not found at {result_subdir}")
    iptm_score = "None"
    if "iptm" in ranking_results:
        iptm_score = ranking_results['iptm'].get(best_model, None)

    if os.path.exists(f"{result_subdir}/pae_{best_model}.json"):
        pae_results = json.load(
            open(f"{result_subdir}/pae_{best_model}.json"))[0]['predicted_aligned_error']
        pae_mtx = np.array(pae_results)

    if iptm_score == "None":
        try:
            check_dict = pickle.load(
                open(os.path.join(result_subdir, f"result_{best_model}.pkl"), 'rb'))
        except FileNotFoundError:
            logging.info(os.path.join(
                result_subdir, f"result_{best_model}.pkl")+" does not exist. Will search for pkl.gz")
            check_dict = pickle.load(gzip.open(os.path.join(
                result_subdir, f"result_{best_model}.pkl.gz"), 'rb'))
        finally:
            logging.info(f"finished reading results for the best model.")
            pae_mtx = check_dict['predicted_aligned_error']
            iptm_score = check_dict['iptm']
    return pae_mtx, iptm_score
