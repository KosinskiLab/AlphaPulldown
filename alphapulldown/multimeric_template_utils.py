
import os, logging, csv,sys
from pathlib import Path
from alphafold.data.templates import (
                                      _extract_template_features,
                                      _build_query_to_hit_index_mapping)
from alphafold.data.templates import SingleHitResult
from alphafold.data.mmcif_parsing import ParsingResult
from alphafold.data.parsers import TemplateHit
from alphapulldown.remove_clashes_low_plddt import MmcifChainFiltered
from typing import Optional
import shutil
import numpy as np

def prepare_multimeric_template_meta_info(csv_path:str, mmt_dir:str) -> dict:
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
                assert os.path.exists(os.path.join(mmt_dir,template)), f"Provided {template} cannot be found in {mmt_dir}. Abort"
                if protein not in parsed_dict:
                    parsed_dict[protein] = {
                        template:chain
                    }
            else:
                logging.error(f"Invalid line found in the file {csv_path}: {row}")
                sys.exit()

    return parsed_dict

def obtain_kalign_binary_path() -> Optional[str]:
    assert shutil.which('kalign') is not None, "Could not find kalign in your environment"
    return shutil.which('kalign')


def parse_mmcif_file(file_id:str,mmcif_file:str) -> ParsingResult:
    """
    Args:
    file_id: A string identifier for this file. Should be unique within the
      collection of files being processed.
    mmcif_file: path to the target mmcif file
    
    Returns:
    A ParsingResult object
    """  
    try:
        mmcif_filtered_obj = MmcifChainFiltered(Path(mmcif_file),file_id)
        parsing_result = mmcif_filtered_obj.parsing_result
    except FileNotFoundError as e:
        parsing_result = None
        print(f"{mmcif_file} could not be found")
    
    return parsing_result

def create_template_hit(index:int, name:str,query:str) -> TemplateHit:
    """
    Create the new template hits and mapping. Currently only supports the cases
    where the query sequence and the template sequence are identical
    
    Args:
    index: index of the hit e.g. numberXX of the customised templates
    name: name of the hit e.g. pdbid_CHAIN
    query: query sequence 

    Returns:
    A TemplateHit object in which hit and query sequences are identical
    """
    aligned_cols = len(query)
    sum_probs = None
    hit_sequence = query 
    indices_hit, indices_query = list(range(aligned_cols)),list(range(aligned_cols))
    return TemplateHit(index=index, name=name,aligned_cols = aligned_cols,
                       sum_probs = sum_probs,query = query, hit_sequence = hit_sequence,
                       indices_query = indices_query, indices_hit = indices_hit)

def extract_multimeric_template_features_for_single_chain(
        query_seq:str,
        pdb_id:str,
        chain_id:str,
        mmcif_file:str,
        index:int =1,

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
    hit = create_template_hit(index, name=f"{pdb_id}_{chain_id}", query=query_seq)
    mapping = _build_query_to_hit_index_mapping(hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,query_seq)
    mmcif_parse_result = parse_mmcif_file(pdb_id, mmcif_file)
    if (mmcif_parse_result is not None) and (mmcif_parse_result.mmcif_object is not None):
        try:
            features, realign_warning = _extract_template_features(
                mmcif_object = mmcif_parse_result.mmcif_object,
                pdb_id = pdb_id,
                mapping = mapping,
                template_sequence = query_seq,
                query_sequence = query_seq,
                template_chain_id = chain_id,
                kalign_binary_path = obtain_kalign_binary_path()
            )
            features['template_sum_probs'] = [0]*4
            # add 1 dimension to template_all_atom_positions and replicate 4 times
            features['template_all_atom_positions'] = np.tile(features['template_all_atom_positions'],(4,1,1,1))
            features['template_all_atom_position'] = features['template_all_atom_positions']
            # replicate all_atom_mask 
            features['template_all_atom_mask'] = features['template_all_atom_masks'][np.newaxis,:]
            for k in ['template_sequence','template_domain_names',
                      'template_aatype']:
                features[k] = [features[k]]*4
            return SingleHitResult(features=features, error=None, warning=realign_warning)
        except Exception as e:
            print(f"Failed to extract template features")
            return SingleHitResult(features=None, error=None, warning=None)