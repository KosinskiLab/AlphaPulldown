from Bio.PDB import Structure, PDBParser, MMCIFParser, NeighborSearch, PDBIO, MMCIFIO
from collections import defaultdict
from absl import app, flags
import logging
import copy
from alphafold.data.mmcif_parsing import _get_atom_site_list
from Bio import SeqIO
from Bio.PDB.Polypeptide import three_to_one, one_to_three

class BioStructure:
    """
    Biopython structure with some checks and mapped mmcif chain id to author chain id
    Stores only the first model and only one chain if provided as argument
    Has methods to remove clashes and low pLDDT residues and can save the structure to a file
    """
    DONORS_ACCEPTORS = ['N', 'O', 'S']
    VDW_RADII = defaultdict(lambda: 1.5,
                            {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8, 'CL': 1.75})

    def __init__(self, input_file_path, chain_id=None):
        self.input_file_path = input_file_path
        self.chain_id = chain_id
        self.mmcif_to_author_chain_id = {}
        self.structure = self.to_bio(input_file_path, chain_id)
        self.seqs = self.extract_seqs()
        self.structure_modified = False


    def __eq__(self, other):
        return self.structure == other.structure


    def to_bio(self, input_file_path, chain_id):
        if input_file_path.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
        elif input_file_path.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            logging.error(f"Unknown file format for {input_file_path}. Accepted formats: PDB, CIF")

        structure = parser.get_structure("model", input_file_path)
        parsed_info = parser._mmcif_dict

        # Error if multiple models are found
        if len(structure.child_list) > 1:
            logging.error(f'{len(structure.child_list)} models found in {input_file_path}!')

        for atom in _get_atom_site_list(parsed_info):
          self.mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

        if chain_id:
            # Error if chain_id is not found
            chain_ids = [chain.id for chain in structure[0]]
            if chain_id not in chain_ids:
                logging.error(f"No {chain_id} in {chain_ids} of {input_file_path}!")

            # Create a new structure to hold the specific chain
            new_structure = Structure.Structure("new_model")
            for model in structure:
                new_model = model.__class__(model.id, new_structure)
                new_structure.add(new_model)
                for chain in model:
                    if chain.id == chain_id:
                        # Simply copy the chain instead of building it from scratch
                        new_chain = copy.deepcopy(chain)
                        new_model.add(new_chain)
                        break
            return new_structure

        return structure

    def extract_seqs(self):
        """
        Extract sequences from PDB/CIF file, if SEQRES records are not present,
        extract from atoms
        Return:
            o list of tuples: (chain_id, sequence)
        """
        seqs = []
        # Parsing SEQRES
        template = self.input_file_path
        chain_id = self.chain_id
        if template.endswith('.pdb'):
            format = 'pdb-seqres'
        elif template.endswith('.cif'):
            format = 'cif-seqres'
        else:
            logging.error(f'Unknown file type for {template}!')
        for record in SeqIO.parse(template, format):
            if record.id == chain_id or record.id == self.mmcif_to_author_chain_id[record.id]:
                seqs.append((chain_id, str(record.seq)))

        # Parsing from atoms if SEQRES records are not found
        if len(seqs) == 0:
            logging.warning(f'No SEQRES records found in {template}! Parsing from atoms!')
            model = self.structure[0]
            for chain in model:
                seq_chain = ''
                for resi in chain:
                    try:
                        one_letter = three_to_one(resi.resname)
                        seq_chain += one_letter
                    except KeyError:
                        logging.warning(f'Skipping {resi.resname} with id {resi.id}')
                seqs.append((chain_id, seq_chain))
        return seqs

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
        Returns:
            o truncated_structure - BioPython structure object with clashing residues removed
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

        for residue in clashing_residues:
            chain = residue.get_parent()
            chain.detach_child(residue.id)
        self.structure_modified = True


    def remove_low_plddt(self, plddt_threshold=50):
        """
        Remove residues with pLDDT score below the threshold.
        o plddt_threshold - threshold for pLDDT score (default: 50)
        Returns:
            o truncated_structure - BioPython structure object with low pLDDT residues removed
        """
        model = self.structure[0]
        low_plddt_residues = set()
        for residue in model.get_residues():
            if any(atom.get_bfactor() < plddt_threshold for atom in residue):
                low_plddt_residues.add(residue)

        logging.info(f"Low pLDDT residues: {len(low_plddt_residues)} out of {len(list(model.get_residues()))}")

        for residue in low_plddt_residues:
            chain = residue.get_parent()
            chain.detach_child(residue.id)

        self.structure_modified = True


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

    bio_struct = BioStructure(input_file_path)
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
