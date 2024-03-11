import os
from absl import logging
import csv
import contextlib
import tempfile
@contextlib.contextmanager
def temp_fasta_file(sequence_str):
    """function that create temp file"""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(sequence_str)
        fasta_file.seek(0)
        yield fasta_file.name

def ensure_directory_exists(directory):
    """
    Ensures that a directory exists. If the directory does not exist, it is created.

    Args:
    directory (str): The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

def parse_csv_file(csv_path, fasta_paths, mmt_dir):
    """
    csv_path (str): Path to the text file with descriptions
        features.csv: A coma-separated file with three columns: PROTEIN name, PDB/CIF template, chain ID.
    fasta_paths (str): path to fasta file(s)
    mmt_dir (str): Path to directory with multimeric template mmCIF files

    Returns:
        a list of dictionaries with the following structure:
    [{"protein": protein_name, "sequence" :sequence", templates": [pdb_files], "chains": [chain_id]}, ...]}]
    """
    protein_names = {}
    for fasta_path in fasta_paths:
        if not os.path.isfile(fasta_path):
            logging.error(f"Fasta file {fasta_path} does not exist.")
            raise FileNotFoundError(f"Fasta file {fasta_path} does not exist.")
        for curr_seq, curr_desc in iter_seqs(fasta_paths):
            protein_names[curr_desc] = curr_seq

    parsed_dict = {}
    with open(csv_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if not row or len(row) != 3:
                logging.warning(f"Skipping invalid line in {csv_path}: {row}")
                continue
            protein, template, chain = map(str.strip, row)
            protein = convert_fasta_description_to_protein_name(protein)
            if protein not in protein_names:
                logging.error(f"Protein {protein} from description.csv is not found in the fasta files.")
                continue
            parsed_dict.setdefault(protein, {"protein": protein, "templates": [], "chains": [], "sequence": None})
            parsed_dict[protein]["sequence"] = protein_names[protein]
            parsed_dict[protein]["templates"].append(os.path.join(mmt_dir, template))
            parsed_dict[protein]["chains"].append(chain)

    return list(parsed_dict.values())

def convert_fasta_description_to_protein_name(line):
    line = line.replace(" ", "_")
    unwanted_symbols = ["|", "=", "&", "*", "@", "#", "`", ":", ";", "$", "?"]
    for symbol in unwanted_symbols:
        if symbol in line:
            line = line.replace(symbol, "_")
    if line.startswith(">"):
        return line[1:]  # Remove the '>' at the beginning.
    else:
        return line

def iter_seqs(fasta_fns):
    """
    Generator that yields sequences and descriptions from multiple fasta files.

    Args:
    fasta_fns (list): A list of fasta file paths.

    Yields:
    tuple: A tuple containing a sequence and its corresponding description.
    """
    for fasta_path in fasta_fns:
        with open(fasta_path, "r") as f:
            sequences, descriptions = parse_fasta(f.read())
            for seq, desc in zip(sequences, descriptions):
                yield seq, desc


def make_dir_monomer_dictionary(monomer_objects_dir):
    """
    a function to gather all monomers across different monomer_objects_dir

    args
    monomer_objects_dir: a list of directories where monomer objects are stored, given by FLAGS.monomer_objects_dir
    """
    output_dict = dict()
    for dir in monomer_objects_dir:
        monomers = os.listdir(dir)
        for m in monomers:
            output_dict[m] = dir
    return output_dict

def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.


    Note:
      This function was built upon alhpafold.data.parsers.parse_fasta in order
      to accomodamte naming convention in this package.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(convert_fasta_description_to_protein_name(line))
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions