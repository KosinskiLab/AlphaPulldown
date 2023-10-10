from collections import defaultdict
from absl import app, flags
import logging
import copy
from alphafold.data.mmcif_parsing import parse
from alphafold.common.residue_constants import residue_atoms
from Bio.PDB import Structure, NeighborSearch, PDBIO, MMCIFIO
from Bio.PDB.Polypeptide import protein_letters_3to1


class MmcifChainFiltered:
    """
    Takes only one chain from the mmcif file
    Has methods to remove clashes and low pLDDT residues and can save the structure to a file
    """
    DONORS_ACCEPTORS = ['N', 'O', 'S']
    VDW_RADII = defaultdict(lambda: 1.5,
                            {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8, 'CL': 1.75})

    def __init__(self, input_file_path, code, chain_id=None):
        self.input_file_path = input_file_path
        self.chain_id = chain_id
        with open(input_file_path) as f:
            mmcif = f.read()
        parsing_result = parse(file_id=code, mmcif_string=mmcif)
        if parsing_result.errors:
            raise Exception(f"Can't parse mmcif file {input_file_path}: {parsing_result.errors}")
        mmcif_object = parse(file_id=code, mmcif_string=mmcif).mmcif_object
        self.sequence_seqres = mmcif_object.chain_to_seqres[chain_id]
        self.seqres_to_structure = mmcif_object.seqres_to_structure[chain_id]
        self.structure, self.sequence_atom = self.extract_chain(mmcif_object.structure, chain_id)
        self.structure_modified = False


    def __eq__(self, other):
        return self.structure == other.structure

    def extract_atom_site_label_seq_id(self):
        """
        Extracts residue index for atoms.
        """
        atoms_label_seq_id = []
        for label_id, residue in self.seqres_to_structure.items():
            if residue.is_missing:
                continue
            name = residue.name
            number_of_atoms_in_residue = len(residue_atoms[name])
            atoms_label_seq_id += [str(label_id + 1)] * number_of_atoms_in_residue
        return atoms_label_seq_id

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
        print(clashing_residues)
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
        Remove residues from the structure
        and seqres_to_structure (that's enough for atoms_label_seq_id).
        """
        ids = []
        for residue in residues:
            chain = residue.get_parent()
            chain.detach_child(residue.id)
            ids.append(residue.id[1])
        # and from seqres_to_structure
        self.seqres_to_structure = \
            {k: v for k, v in self.seqres_to_structure.items() if
             v.position.residue_number not in ids}

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
