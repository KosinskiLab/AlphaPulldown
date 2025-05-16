#!/usr/bin/env python3
"""
AlphaPulldown modelling setup utilities

- create_uniprot_runner: Jackhmmer runner factory
- parse_fold: parse CLI input into region specs
- create_interactors: build MonomericObject lists via make_monomer_from_range
- load_monomer_objects: load pickled MonomericObject (compressed or not)
- pad_input_features: pad feature arrays to desired sizes
- check_empty_templates/mk_mock_template: ensure template features exist
"""
import lzma
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from absl import logging
from alphafold.data import parsers, templates
from alphafold.data.tools import jackhmmer as jackhmmer_tool
from alphapulldown.builders import make_monomer_from_range, MonomericObject

logging.set_verbosity(logging.INFO)


def create_uniprot_runner(
    jackhmmer_binary_path: str,
    uniprot_database_path: str
) -> jackhmmer_tool.Jackhmmer:
    """
    Create and return a Jackhmmer runner for Uniprot MSA searches.
    """
    return jackhmmer_tool.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniprot_database_path
    )


def parse_fold(
    input_list: List[str],
    delimiter: str = "+"
) -> List[List[Tuple[str, int, int, int]]]:
    """
    Parse entries like 'file.fasta:0:10-50+file2.fasta:1:all' into
    [[(path, chain, start, stop), ...], ...].
    Chain is 0-based; 'all' yields full sequence range.
    Raises ValueError on malformed input or missing files.
    """
    jobs: List[List[Tuple[str,int,int,int]]] = []
    for entry in input_list:
        parts = entry.split(delimiter)
        job: List[Tuple[str,int,int,int]] = []
        for part in parts:
            try:
                fasta, chain_str, region_str = part.split(":")
            except ValueError:
                raise ValueError(f"Invalid fold spec '{part}'")
            # chain idx
            try:
                chain = int(chain_str)
            except ValueError:
                raise ValueError(f"Invalid chain index '{chain_str}' in '{part}'")
            fasta_path = Path(fasta)
            if not fasta_path.exists():
                raise FileNotFoundError(f"FASTA not found: {fasta}")
            seqs, _ = parsers.parse_fasta(fasta_path.read_text())
            if chain < 0 or chain >= len(seqs):
                raise IndexError(f"Chain index {chain} out of bounds for {fasta}")
            full_seq = seqs[chain]
            if region_str.lower() == 'all':
                start, stop = 1, len(full_seq)
            else:
                try:
                    start_s, stop_s = region_str.split("-")
                    start, stop = int(start_s), int(stop_s)
                except Exception:
                    raise ValueError(f"Invalid region '{region_str}' in '{part}'")
            job.append((str(fasta_path), chain, start, stop))
        jobs.append(job)
    return jobs


def create_interactors(
    jobs: List[List[Tuple[str, int, int, int]]],
    jackhmmer_binary_path: str,
    uniprot_database_path: str
) -> List[List[MonomericObject]]:
    """
    From parsed jobs, return lists of MonomericObject for each job.
    Uses a single uniprot_runner for all.
    """
    runner = create_uniprot_runner(jackhmmer_binary_path, uniprot_database_path)
    all_objects: List[List[MonomericObject]] = []
    for job in jobs:
        objs: List[MonomericObject] = []
        for fasta, chain, start, stop in job:
            mono = make_monomer_from_range(fasta, chain, start, stop, runner)
            objs.append(mono)
        all_objects.append(objs)
    return all_objects

def create_custom_info(all_proteins : List[List[Dict[str, str]]]) -> List[Dict[str, List[str]]]:
    """
    Create a dictionary representation of data for a custom input file.

    Parameters
    ----------
    all_proteins : List[List[Dict[str, str]]]
       A list of lists of protein names or sequences. Each element
       of the list is a nother list of dictionaries thats should be included in the data.

    Returns
    -------
     List[Dict[str, List[str]]]
        A list of dictionaries. Within each dictionary: each key is a column name following the
        pattern 'col_X' where X is the column index starting from 1.
        Each key maps to a list containing a single protein name or
        sequence from the input list.

    """
    output = []
    def process_single_dictionary(all_proteins):
        num_cols = len(all_proteins)
        data = dict()
        for i in range(num_cols):
            data[f"col_{i + 1}"] = [all_proteins[i]]
        return data
    for i in all_proteins:
        curr_data = process_single_dictionary(i)
        output.append(curr_data)
    return output

def load_monomer_objects(
    monomer_dir_dict: Dict[str,str],
    protein_name: str
) -> MonomericObject:
    """
    Load a MonomericObject from .pkl or .pkl.xz in given dirs.
    """
    if f"{protein_name}.pkl" in monomer_dir_dict:
        base = monomer_dir_dict[f"{protein_name}.pkl"]
        target = Path(base) / f"{protein_name}.pkl"
    elif f"{protein_name}.pkl.xz" in monomer_dir_dict:
        base = monomer_dir_dict[f"{protein_name}.pkl.xz"]
        target = Path(base) / f"{protein_name}.pkl.xz"
    else:
        raise FileNotFoundError(f"No file found for {protein_name}")
    if target.suffix == '.xz':
        with lzma.open(target, 'rb') as f:
            monomer = pickle.load(f)
    else:
        with open(target, 'rb') as f:
            monomer = pickle.load(f)
    return monomer


def pad_input_features(
    feature_dict: dict,
    desired_num_res: int,
    desired_num_msa: int
) -> None:
    """
    Pad feature arrays to desired residue and MSA counts in-place.
    """
    def pad_array(v: np.ndarray, axes: List[int], widths: List[int]) -> np.ndarray:
        pad = [(0,0)] * v.ndim
        for ax, w in zip(axes, widths):
            pad[ax] = (0, w)
        return np.pad(v, pad)

    assembly = feature_dict.pop('assembly_num_chains')
    templates_n = feature_dict.pop('num_templates')
    feature_dict.pop('seq_length', None)
    feature_dict.pop('num_alignments', None)

    msa_mat = feature_dict['msa']
    orig_msa, orig_res = msa_mat.shape
    pad_msa = max(0, desired_num_msa - orig_msa)
    pad_res = max(0, desired_num_res - orig_res)

    for k, v in list(feature_dict.items()):
        if not isinstance(v, np.ndarray):
            continue
        axes: List[int] = []
        widths: List[int] = []
        if orig_msa in v.shape:
            axes.append(v.shape.index(orig_msa)); widths.append(pad_msa)
        if orig_res in v.shape:
            axes.append(v.shape.index(orig_res)); widths.append(pad_res)
        if axes:
            feature_dict[k] = pad_array(v, axes, widths)

    feature_dict['seq_length'] = np.array([desired_num_res])
    feature_dict['num_alignments'] = np.array([desired_num_msa])
    feature_dict['assembly_num_chains'] = assembly
    feature_dict['num_templates'] = templates_n


def check_empty_templates(feature_dict: dict) -> bool:
    """Return True if template arrays are empty."""
    key = 'template_all_atom_masks'
    if key in feature_dict:
        return feature_dict[key].size == 0 or feature_dict['template_aatype'].size == 0
    return False


def mk_mock_template(feature_dict: dict) -> dict:
    """Inject a dummy template if none exist."""
    ln = feature_dict['aatype'].shape[0]
    num_temp = 1
    pos = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
    mask = np.zeros((ln, templates.residue_constants.atom_type_num))
    seq = 'A' * ln
    aatype = templates.residue_constants.sequence_to_onehot(
        seq, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    feature_dict.update({
        'template_all_atom_positions': pos[None,...].repeat(num_temp,0),
        'template_all_atom_masks': mask[None,...].repeat(num_temp,0),
        'template_sequence': [b'none']*num_temp,
        'template_aatype': aatype[None,...].repeat(num_temp,0),
        'template_domain_names': [b'none']*num_temp,
        'template_sum_probs': np.zeros(num_temp, dtype=np.float32)
    })
    return feature_dict
