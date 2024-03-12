
from absl import logging
logging.set_verbosity(logging.INFO)
import os
import csv
import sys
from pathlib import Path
from alphafold.data.templates import (
    _extract_template_features,
    _build_query_to_hit_index_mapping)
from alphafold.data.templates import SingleHitResult
from alphafold.data.mmcif_parsing import ParsingResult
from alphapulldown.utils.remove_clashes_low_plddt import MmcifChainFiltered
from typing import Optional, Dict
import shutil
import numpy as np

from alphafold.data.tools import kalign
from alphafold.data import parsers


def prepare_multimeric_template_meta_info(csv_path: str, mmt_dir: str) -> dict:
    """
    Adapted from https://github.com/KosinskiLab/AlphaPulldown/blob/231863af7faa61fa04d45829c90a3bab9d9e2ff2/alphapulldown/create_individual_features_with_templates.py#L107C1-L159C38
    by @DimaMolod

    Args:
    csv_path: Path to the text file with descriptions
        features.csv: A coma-separated file with three columns: PROTEIN name, PDB/CIF template, chain ID.
    mmt_dir: Path to directory with multimeric template mmCIF files

    Returns:
        a list of dictionaries with the following structure:
    [{"protein": protein_name, "sequence" :sequence", templates": [pdb_files], "chains": [chain_id]}, ...]}]
    """
    # Parse csv file
    parsed_dict = {}
    with open(csv_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # skip empty lines
            if not row:
                continue
            if len(row) == 3:
                protein, template, chain = [item.strip() for item in row]
                assert os.path.exists(os.path.join(
                    mmt_dir, template)), f"Provided {template} cannot be found in {mmt_dir}. Abort"
                if protein not in parsed_dict:
                    parsed_dict[protein] = {
                        template: chain
                    }
            else:
                logging.error(
                    f"Invalid line found in the file {csv_path}: {row}")
                sys.exit()

    return parsed_dict


def obtain_kalign_binary_path() -> Optional[str]:
    assert shutil.which(
        'kalign') is not None, "Could not find kalign in your environment"
    return shutil.which('kalign')


def parse_mmcif_file(file_id: str, mmcif_file: str, chain_id: str) -> ParsingResult:
    """
    Args:
    file_id: A string identifier for this file. Should be unique within the
      collection of files being processed.
    mmcif_file: path to the target mmcif file

    Returns:
    A ParsingResult object
    """
    try:
        mmcif_filtered_obj = MmcifChainFiltered(
            Path(mmcif_file), file_id, chain_id=chain_id)
        parsing_result = mmcif_filtered_obj.parsing_result
    except FileNotFoundError as e:
        parsing_result = None
        print(f"{mmcif_file} could not be found")

    return parsing_result


def _obtain_mapping(mmcif_parse_result: ParsingResult, chain_id: str,
                    original_query_sequence: str) -> Dict[int,int]:
    """
    A function to obtain teh mapping between query sequence and the selected
    chain from customerised template(a pdb or mmcif file)

    Args:
    mmcif_parse_result: a ParsingResult object from parse_mmcif_file() function
    chain_id: id of the chain to use from the mmcif/pdb file
    original_query_sequence: original protein sequence that to be modelled, without gaps

    Return:
    mapping: a dictionary {int:int} that maps original query sequence to chain sequence in the PDB/mmcif file
    e.g. if query sequence is ACDES and chain sequence is MHADE then mapping will be 
    {0:2, 2:3, 3:4}
    """
    mmcif_object = mmcif_parse_result.mmcif_object
    parsed_resseq = mmcif_object.chain_to_seqres[chain_id]
    aligner = kalign.Kalign(binary_path = obtain_kalign_binary_path())
    parsed_a3m = parsers.parse_a3m(aligner.align([original_query_sequence,parsed_resseq]))
    aligned_query_sequence, aligned_template_hit = parsed_a3m.sequences
    hit_indices = parsers._get_indices(aligned_template_hit, start=0)
    query_indecies = parsers._get_indices(
                    aligned_query_sequence, start=0)

    mapping = _build_query_to_hit_index_mapping(aligned_query_sequence,
                                                aligned_template_hit,
                                                hit_indices,
                                                query_indecies, original_query_sequence)
    return mapping,parsed_resseq


def extract_multimeric_template_features_for_single_chain(
        query_seq: str,
        pdb_id: str,
        chain_id: str,
        mmcif_file: str,
) -> SingleHitResult:
    """
    Args:
    index: index of the hit e.g. numberXX of the customised templates
    query_seq: the sequence to be modelled, single chain
    pdb_id: the id of the PDB file or the name of the pdb file where the multimeric template structure is written
    chain_id: which chain of the multimeric template that this query sequence will be aligned to 
    mmcif_file: path to the .cif file that is going to be parsed.

    Returns:
    A SingleHitResult object
    """
    mmcif_parse_result = parse_mmcif_file(
        pdb_id, mmcif_file, chain_id=chain_id)
    if (mmcif_parse_result is not None) and (mmcif_parse_result.mmcif_object is not None):
        mapping,template_sequence = _obtain_mapping(mmcif_parse_result=mmcif_parse_result,
                                  chain_id=chain_id,
                                  original_query_sequence=query_seq)

        try:
            features, realign_warning = _extract_template_features(
                mmcif_object=mmcif_parse_result.mmcif_object,
                pdb_id=pdb_id,
                mapping=mapping,
                template_sequence=template_sequence,
                query_sequence=query_seq,
                template_chain_id=chain_id,
                kalign_binary_path=obtain_kalign_binary_path()
            )
        except Exception as e:
            logging.warning(f"Failed to extract template features")
            return SingleHitResult(features=None, error=None, warning=None)
        try:
            features['template_sum_probs'] = [0]*4
            # add 1 dimension to template_all_atom_positions and replicate 4 times
            features['template_all_atom_positions'] = np.tile(
                features['template_all_atom_positions'], (4, 1, 1, 1))
            features['template_all_atom_position'] = features['template_all_atom_positions']
            # replicate all_atom_mask
            features['template_all_atom_mask'] = features['template_all_atom_masks'][np.newaxis, :]
            for k in ['template_sequence', 'template_domain_names',
                      'template_aatype']:
                features[k] = [features[k]]*4
            return SingleHitResult(features=features, error=None, warning=realign_warning)
        except Exception as e:
            logging.warning("Failed to construct SingleHitResult")
