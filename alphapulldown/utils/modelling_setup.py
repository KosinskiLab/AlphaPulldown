#!/usr/bin/env python3
"""
Utility functions for AlphaPulldown workflows:
- parse_fold: parse CLI input into region specs
- create_interactors: build MonomericObject lists via make_monomer_from_range
- create_uniprot_runner: Jackhmmer runner factory
- pad_input_features: pad feature arrays if needed
- check_empty_templates / mk_mock_template: ensure templates present
"""
import os
import lzma
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from absl import logging
from alphafold.data import parsers, templates
from alphapulldown.objects import make_monomer_from_range, MonomericObject
from alphapulldown.utils.modelling_setup import create_uniprot_runner

logging.set_verbosity(logging.INFO)


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
    jobs = []
    for entry in input_list:
        parts = entry.split(delimiter)
        job = []
        for part in parts:
            try:
                fasta, chain_str, region_str = part.split(":")
            except ValueError:
                raise ValueError(f"Invalid fold spec '{part}'")
            # chain index
            chain = int(chain_str)
            # region
            if region_str.lower() == 'all':
                content = Path(fasta).read_text()
                seqs, _ = parsers.parse_fasta(content)
                seq = seqs[chain]
                start, stop = 1, len(seq)
            else:
                try:
                    start_s, stop_s = region_str.split("-")
                    start, stop = int(start_s), int(stop_s)
                except Exception:
                    raise ValueError(f"Invalid region '{region_str}' in '{part}'")
            if not Path(fasta).exists():
                raise FileNotFoundError(f"FASTA not found: {fasta}")
            job.append((fasta, chain, start, stop))
        jobs.append(job)
    return jobs


def create_interactors(
    jobs: List[List[Tuple[str, int, int, int]]],
    jackhmmer_binary: str,
    uniprot_db: str
) -> List[List[MonomericObject]]:
    """
    From parsed jobs, return lists of MonomericObject for each job.
    Uses a single uniprot_runner for all.
    """
    runner = create_uniprot_runner(jackhmmer_binary, uniprot_db)
    all_objects: List[List[MonomericObject]] = []
    for job in jobs:
        objs: List[MonomericObject] = []
        for fasta, chain, start, stop in job:
            mono = make_monomer_from_range(fasta, chain, start, stop, runner)
            objs.append(mono)
        all_objects.append(objs)
    return all_objects


def create_uniprot_runner(jackhmmer_binary: str, uniprot_database_path: str):
    """Alias to Jackhmmer runner factory."""
    return create_uniprot_runner(jackhmmer_binary, uniprot_database_path)


def pad_input_features(
    feature_dict: dict,
    desired_num_res: int,
    desired_num_msa: int
) -> None:
    """
    Pad feature arrays to desired residue and MSA counts in-place.
    """
    def pad_array(v: np.ndarray, pad_axes: list, pad_widths: list):
        pw = [(0,0)] * v.ndim
        for ax, w in zip(pad_axes, pad_widths):
            pw[ax] = (0, w)
        return np.pad(v, pad_width=pw)

    assembly_num_chains = feature_dict.pop('assembly_num_chains')
    num_templates = feature_dict.pop('num_templates')
    feature_dict.pop('seq_length', None)
    feature_dict.pop('num_alignments', None)

    msa_mat = feature_dict['msa']
    orig_msa, orig_res = msa_mat.shape
    pad_msa = max(0, desired_num_msa - orig_msa)
    pad_res = max(0, desired_num_res - orig_res)

    for k, v in list(feature_dict.items()):
        if not isinstance(v, np.ndarray):
            continue
        axes = []
        widths = []
        if v.shape and orig_msa in v.shape:
            axes.append(v.shape.index(orig_msa)); widths.append(pad_msa)
        if v.shape and orig_res in v.shape:
            axes.append(v.shape.index(orig_res)); widths.append(pad_res)
        if axes:
            feature_dict[k] = pad_array(v, axes, widths)

    feature_dict['seq_length'] = np.array([desired_num_res])
    feature_dict['num_alignments'] = np.array([desired_num_msa])
    feature_dict['assembly_num_chains'] = assembly_num_chains
    feature_dict['num_templates'] = num_templates


def check_empty_templates(feature_dict: dict) -> bool:
    """Return True if template arrays are empty."""
    mask_key = 'template_all_atom_masks'
    if mask_key in feature_dict:
        return feature_dict[mask_key].size == 0 or feature_dict['template_aatype'].size == 0
    return False


def mk_mock_template(feature_dict: dict) -> dict:
    """Inject a dummy template if none exist."""
    ln = feature_dict['aatype'].shape[0]
    num_temp = 1
    pos = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
    mask = np.zeros((ln, templates.residue_constants.atom_type_num))
    seq = 'A'*ln
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
