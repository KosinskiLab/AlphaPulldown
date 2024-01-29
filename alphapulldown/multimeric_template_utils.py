
from alphafold.data.templates import _read_file, _extract_template_features,_build_query_to_hit_index_mapping
from alphafold.data.templates import SingleHitResult
from alphafold.data import mmcif_parsing
from alphafold.data.mmcif_parsing import ParsingResult
from alphafold.data.parsers import TemplateHit
from typing import Optional
import shutil


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
        mmcif_string = _read_file(mmcif_file)
        parsing_result = mmcif_parsing.parse(file_id = file_id,mmcif_string = mmcif_string)
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

def exctract_multimeric_template_features_for_single_chain(
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
        mmcif_chain_seq_map = mmcif_parse_result.mmcif_object.chain_to_seqres
        try:
            template_seq = mmcif_chain_seq_map[chain_id]
        except Exception as e:
            print(f"chain: {chain_id} does not exist in {mmcif_file}. Please double check you input.")
        try:
            features, realign_warning = _extract_template_features(
                mmcif_object = mmcif_parse_result.mmcif_object,
                pdb_id = pdb_id,
                mapping = mapping,
                template_seq = template_seq,
                query_sequence = query_seq,
                template_chain_id = chain_id,
                kalign_binary_path = obtain_kalign_binary_path()
            )
            features['template_sum_probs'] = [0]
            return SingleHitResult(features=features, error=None, warning=realign_warning)
        except Exception as e:
            print(f"Failed to extract template features")
            return SingleHitResult(features=None, error=None, warning=None)