
from alphafold.data.templates import _read_file, _extract_template_features
from alphafold.data import mmcif_parsing
from alphafold.data.mmcif_parsing import ParsingResult

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

def create_template_hits():
    """Create the new template hits and mapping """

def exctract_multimeric_template_features_for_single_chain(
        query_seq:str,
        pdb_id:str,
        chain_id:str,
        mmcif_file:str,

):
    """
    Now implementing _process_single_hit()
    Args:
    query_seq: the sequence to be modelled, single chain
    pdb_id: the id of the PDB file or the name of the pdb file where 
    the multimeric template structure is written
    chain_id: which chain of the multimeric template that this query sequence 
    will be aligned to 
    mmcif_file: path to the .cif file that is going to be parsed.
    """

    mmcif_parse_result = parse_mmcif_file(pdb_id, mmcif_file)
    if (mmcif_parse_result is not None) and (mmcif_parse_result.mmcif_object is not None):
        #Now implementing _extract_template_features here#
        mmcif_chain_seq_map = mmcif_parse_result.mmcif_object.chain_to_seqres
        try:
            template_seq = mmcif_chain_seq_map[chain_id]
        except Exception as e:
            print(f"chain: {chain_id} does not exist in {mmcif_file}. Please double check you input.")
        try:
            features, realign_warning = _extract_template_features(
                mmcif_object = mmcif_parse_result.mmcif_object,
                pdb_id = pdb_id,

            )