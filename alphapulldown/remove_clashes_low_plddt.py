from collections import defaultdict
from absl import app, flags
import logging
import copy
from alphafold.data.mmcif_parsing import parse
from alphafold.common.residue_constants import residue_atoms, atom_types
from Bio.PDB import NeighborSearch, PDBIO, MMCIFIO
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio import SeqIO
from colabfold.batch import convert_pdb_to_mmcif
from Bio.PDB import Structure, Model, Chain, Residue
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

STANDARD_AMINO_ACIDS = residue_atoms.keys()


def extract_seqs(template, chain_id):
    """
    Extract sequences from PDB/CIF file using Bio.SeqIO.
    o input_file_path - path to the input file
    o chain_id - chain ID
    Returns:
        o sequence_atom - sequence from ATOM records
        o sequence_seqres - sequence from SEQRES records
    """

    file_type = template.suffix.lower()
    if template.suffix.lower() != '.pdb' and template.suffix.lower() != '.cif':
        raise ValueError(f"Unknown file type for {template}!")

    format_types = [f"{file_type[1:]}-atom", f"{file_type[1:]}-seqres"]
    # initialize the sequences
    sequence_atom = None
    sequence_seqres = None
    # parse
    for format_type in format_types:
        for record in SeqIO.parse(template, format_type):
            chain = record.annotations['chain']
            if chain == chain_id:
                if format_type.endswith('atom'):
                    sequence_atom = str(record.seq)
                elif format_type.endswith('seqres'):
                    sequence_seqres = str(record.seq)
    if sequence_atom is None:
        logging.error(f"No atom sequence found for chain {chain_id}")
    if sequence_seqres is None:
        logging.warning(f"No SEQRES sequence found for chain {chain_id}")
    return sequence_atom, sequence_seqres


def remove_hydrogens_and_irregularities(structure):
    """
    Takes a BioPython Structure object and returns a new Structure object without hydrogen atoms,
    alternative atom locations, and non-standard amino acids.
    """
    new_structure = Structure.Structure(structure.id)

    for model in structure:
        new_model = Model.Model(model.id)
        new_structure.add(new_model)

        for chain in model:
            new_chain = Chain.Chain(chain.id)
            new_model.add(new_chain)

            for residue in chain:
                # Check for HETATM and standard amino acid
                if residue.id[0] == 'H' or residue.resname.strip() not in STANDARD_AMINO_ACIDS:
                    continue

                new_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
                new_chain.add(new_residue)

                for atom in residue:
                    # Check for hydrogen atoms and alternative locations
                    if atom.name in atom_types: # != 'H' and atom.get_name() != 'OXT':
                        if atom.altloc != 'A':
                            new_residue.add(atom)

    return new_structure


class MmcifChainFiltered:
    """
    Takes only one chain from the mmcif file
    Has methods to remove clashes and low pLDDT residues and can save the structure to a file
    """
    DONORS_ACCEPTORS = ['N', 'O', 'S']
    VDW_RADII = defaultdict(lambda: 1.5,
                            {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8, 'CL': 1.75})


    def __init__(self, input_file_path, code, chain_id=None):
        self.atom_site_label_seq_id = None
        self.atom_to_label_id = None
        self.atom_site_label_seq_ids = None
        self.input_file_path = input_file_path
        self.chain_id = chain_id
        logging.info("Parsing SEQRES...")
        sequence_atom, sequence_seqres = extract_seqs(input_file_path, chain_id)
        if not sequence_seqres:
            logging.warning(f"No SEQRES was found in {input_file_path}! Parsing from atoms...")
        file_type = input_file_path.suffix.lower()
        if file_type == ".pdb":
            logging.info(f"Converting to mmCIF: {input_file_path}")
            convert_pdb_to_mmcif(input_file_path)
            self.input_file_path = input_file_path.with_suffix(".cif")
        with open(self.input_file_path) as f:
            mmcif = f.read()
        parsing_result = parse(file_id=code, mmcif_string=mmcif)
        if parsing_result.errors:
            raise Exception(f"Can't parse mmcif file {self.input_file_path}: {parsing_result.errors}")
        mmcif_object = parsing_result.mmcif_object
        self.seqres_to_structure = mmcif_object.seqres_to_structure[chain_id]
        structure, sequence_atom = self.extract_chain(mmcif_object.structure, chain_id)
        self.sequence_atom = sequence_atom
        self.sequence_seqres = mmcif_object.chain_to_seqres[chain_id]
        if self.sequence_seqres is None:
            self.sequence_seqres = sequence_seqres
        if self.sequence_atom is None:
            self.sequence_atom = sequence_atom
        self.structure = remove_hydrogens_and_irregularities(structure)
        self.map_atoms_to_label_seq_id()
        self.extract_atom_site_label_seq_id()
        self.structure_modified = False


    def __eq__(self, other):
        return self.structure == other.structure


    def map_atoms_to_label_seq_id(self):
        """
        Maps structure atoms to label seq for a particular chain of an mmCIF file.
        """
        mmcif_file = self.input_file_path
        mmcif_dict = MMCIF2Dict(mmcif_file)

        # Extract label_seq_id, chain_id, and residue number (auth_seq_id)
        label_seq_ids = mmcif_dict.get('_atom_site.label_seq_id')
        chain_ids = mmcif_dict.get('_atom_site.auth_asym_id')
        residue_numbers = mmcif_dict.get('_atom_site.auth_seq_id')

        if not (label_seq_ids and chain_ids and residue_numbers):
            raise Exception(f"Error: Required data not found in mmCIF file {mmcif_file}")

        # Pre-process MMCIF data into a map
        mmcif_map = {(chain_id, residue_number): label_id
                     for label_id, chain_id, residue_number in zip(label_seq_ids, chain_ids, residue_numbers)}

        # Initialize the list of dictionaries
        atom_to_labels = []

        # Filter by target chain ID and map atoms to label_seq_id
        target_chain = self.chain_id
        for atom in self.structure.get_atoms():
            atom_chain_id = atom.get_parent().get_parent().id
            atom_residue_number = atom.get_parent().id[1]

            if atom_chain_id == target_chain and (atom_chain_id, str(atom_residue_number)) in mmcif_map:
                label_id = mmcif_map[(atom_chain_id, str(atom_residue_number))]
                atom_to_label = {'atom': atom, 'sequence_id': label_id, 'is_missing': False}
                atom_to_labels.append(atom_to_label)
        self.atom_to_label_id = atom_to_labels


    def extract_atom_site_label_seq_id(self):
        """
        Extracts residue indicies for atoms.
        """
        atoms_label_seq_ids = []
        for atom_to_label in self.atom_to_label_id:
            if not atom_to_label['is_missing']:
                atoms_label_seq_ids.append(atom_to_label['sequence_id'])
        number_of_labels = len(atoms_label_seq_ids)
        number_of_atoms = sum(1 for _ in self.structure.get_atoms())
        if number_of_labels == number_of_atoms:
            len_seq = len(self.sequence_seqres or self.sequence_atom)
            ids = [int(f) for f in atoms_label_seq_ids]
            max_id = max(ids)
            min_id = min(ids)
            if len_seq < max_id:
                logging.warning(f"Max sequence id {max_id} refer to non-existent residues! Resetting label ids...")
                self.atom_site_label_seq_ids = None
            elif  min_id < 1:
                logging.warning(f"Min sequence id {min_id} is less than 1! Resetting label ids...")
                self.atom_site_label_seq_ids = None
            else:
                self.atom_site_label_seq_ids = atoms_label_seq_ids
        else:
            raise Exception(f"Number of sequence ids {number_of_labels} "
                            f"is not equal to number of atoms in structure {number_of_atoms}")


    def extract_chain(self, model, chain_id):
        """
        Extracts a chain and parses sequence from atoms.
        """

        # The author chain ids
        chain_ids = [chain.id for chain in model]
        if chain_id not in chain_ids:
            raise ValueError(f"No {chain_id} in author {chain_ids} of {self.input_file_path}!")

        new_structure = Structure.Structure(f"struct_{chain_id}")
        new_model = model.__class__(model.id, new_structure)
        new_structure.add(new_model)
        resis = list(protein_letters_3to1.keys())
        residues_to_remove = []

        for chain in model:
            if chain.id == chain_id:
                seq = ''
                for resi in chain:
                    if resi.resname not in resis:
                        residues_to_remove.append(resi)
                        continue
                    try:
                        one_letter = protein_letters_3to1[resi.resname]
                        seq += one_letter
                    except KeyError:
                        logging.info(f'Skipping residue {resi.resname} with id {resi.id}, chain {chain_id}')

                for resi in residues_to_remove:
                    chain.detach_child(resi.id)

                new_chain = copy.deepcopy(chain)
                new_model.add(new_chain)

        return new_structure, seq


    def is_potential_hbond(self, atom1, atom2):
        """
        Check if two atoms are potential hydrogen bond donors and acceptors.
        """
        if (atom1.element in self.DONORS_ACCEPTORS and atom2.element in self.DONORS_ACCEPTORS) or \
                (atom2.element in self.DONORS_ACCEPTORS and atom1.element in self.DONORS_ACCEPTORS):
            return True
        return False


    def remove_clashes(self, threshold=0.9, hb_allowance=0.4):
        """
        Remove residues that are clashing with other residues in the structure.
        o threshold - threshold for VDW overlap to identify clashes (default: 0.9)
        o hb_allowance - correction to threshold for hydrogen bonding (default: 0.4)
        """
        model = self.structure[0]
        ns = NeighborSearch(list(model.get_atoms()))
        clashing_atoms = set()

        for residue in model.get_residues():
            for atom in residue:
                neighbors = ns.search(atom.get_coord(), self.VDW_RADII[atom.element] + max(self.VDW_RADII.values()))
                for neighbor in neighbors:
                    if neighbor.get_parent() == atom.get_parent() or \
                            abs(neighbor.get_parent().id[1] - atom.get_parent().id[1]) <= 1:
                        continue
                    overlap = (self.VDW_RADII[atom.element] + self.VDW_RADII[neighbor.element]) - (atom - neighbor)
                    if self.is_potential_hbond(atom, neighbor):
                        overlap -= hb_allowance
                    if overlap >= threshold:
                        clashing_atoms.add(atom)
                        clashing_atoms.add(neighbor)
                        break

        # Remove residues if at least one atom is clashing
        clashing_residues = set(atom.get_parent() for atom in clashing_atoms)
        logging.info(f"Unique clashing atoms: {len(clashing_atoms)} out of {len(list(model.get_atoms()))}")
        logging.info(f"Unique clashing residues: {len(clashing_residues)} out of {len(list(model.get_residues()))}")
        # remove from structure and seqres_to_structure
        if len(clashing_residues) > 0:
            self.remove_residues(clashing_residues)
        self.structure_modified = True


    def remove_low_plddt(self, plddt_threshold=50):
        """
        Remove residues with pLDDT score below the threshold.
        o plddt_threshold - threshold for pLDDT score (default: 50)
        """
        model = self.structure[0]
        low_plddt_residues = set()
        for residue in model.get_residues():
            if any(atom.get_bfactor() < plddt_threshold for atom in residue):
                low_plddt_residues.add(residue)

        logging.info(f"Low pLDDT residues: {len(low_plddt_residues)} out of {len(list(model.get_residues()))}")
        # remove from structure and seqres_to_structure
        if len(low_plddt_residues) > 0:
            self.remove_residues(low_plddt_residues)
        self.structure_modified = True


    def remove_residues(self, residues):
        """
        Remove residues from the structure and modify atom_to_label_id dict
        """
        # Remove from structure
        for residue in residues:
            chain = residue.get_parent()
            chain.detach_child(residue.id)

        # Change is_missing to True for atom_to_label_id list
        for atom_dict in self.atom_to_label_id:
            if atom_dict['atom'].get_parent() in residues:
                atom_dict['is_missing'] = True

        # Update atom site label seq id
        self.extract_atom_site_label_seq_id()


    def save_structure(self, output_file_path):
        """
        Save structure to a file.
        """
        if output_file_path.endswith(".pdb"):
            io = PDBIO()
        elif output_file_path.endswith(".cif"):
            io = MMCIFIO()
        io.set_structure(self.structure)
        io.save(output_file_path)


def main(argv):
    input_file_path = flags.FLAGS.input_file_path
    output_file_path = flags.FLAGS.output_file_path
    threshold = flags.FLAGS.threshold
    hb_allowance = flags.FLAGS.hb_allowance
    plddt_threshold = flags.FLAGS.plddt_threshold

    bio_struct = MmcifChainFiltered(input_file_path, "TEST")
    bio_struct.remove_clashes(threshold, hb_allowance)
    bio_struct.remove_low_plddt(plddt_threshold)

    if output_file_path:
        bio_struct.save_structure(output_file_path)


if __name__ == '__main__':
    flags.DEFINE_string("input_file_path", None, "Path to the input PDB or CIF file")
    flags.DEFINE_string("output_file_path", None, "Path to save the output file. Optional if --save is False")
    flags.DEFINE_float("threshold", 0.9, "Threshold for VDW overlap to identify clashes")
    flags.DEFINE_float("hb_allowance", 0.4, "Allowance for hydrogen bonding (default: 0.0)")
    flags.DEFINE_float("plddt_threshold", 50, "Threshold for pLDDT score (default: 50)")
    flags.mark_flags_as_required(["input_file_path"])
    app.run(main)
