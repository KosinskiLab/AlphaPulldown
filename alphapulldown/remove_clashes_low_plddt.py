from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from collections import defaultdict
from absl import app, flags
import logging

DONORS_ACCEPTORS = ['N', 'O', 'S']
VDW_RADII = defaultdict(lambda: 1.5)
VDW_RADII.update({'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8, 'CL': 1.75})


def is_potential_hbond(atom1, atom2):
    """
    Check if two atoms are potential hydrogen bond donors and acceptors.
    """
    if (atom1.element in DONORS_ACCEPTORS and atom2.element in DONORS_ACCEPTORS) or \
            (atom2.element in DONORS_ACCEPTORS and atom1.element in DONORS_ACCEPTORS):
        return True
    return False


def remove_clashes(structure, threshold=0.9, hb_allowance=0.4):
    """
    Remove residues that are clashing with other residues in the structure.
    o structure - BioPython structure object
    o threshold - threshold for VDW overlap to identify clashes (default: 0.9)
    o hb_allowance - correction to threshold for hydrogen bonding (default: 0.4)
    Returns:
        o truncated_structure - BioPython structure object with clashing residues removed
    """
    model = structure[0]
    ns = NeighborSearch(list(model.get_atoms()))
    clashing_atoms = set()

    for residue in model.get_residues():
        for atom in residue:
            neighbors = ns.search(atom.get_coord(), VDW_RADII[atom.element] + max(VDW_RADII.values()))
            for neighbor in neighbors:
                if neighbor.get_parent() == atom.get_parent() or \
                        abs(neighbor.get_parent().id[1] - atom.get_parent().id[1]) <= 1:
                    continue
                overlap = (VDW_RADII[atom.element] + VDW_RADII[neighbor.element]) - (atom - neighbor)
                if is_potential_hbond(atom, neighbor):
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

    return structure


def remove_low_plddt(structure, plddt_threshold=50):
    """
    Remove residues with pLDDT score below the threshold.
    o structure - BioPython structure object
    o plddt_threshold - threshold for pLDDT score (default: 50)
    Returns:
        o truncated_structure - BioPython structure object with low pLDDT residues removed
    """
    model = structure[0]
    low_plddt_residues = set()
    for residue in model.get_residues():
        if any(atom.get_bfactor() < plddt_threshold for atom in residue):
            low_plddt_residues.add(residue)

    logging.info(f"Low pLDDT residues: {len(low_plddt_residues)} out of {len(list(model.get_residues()))}")

    for residue in low_plddt_residues:
        chain = residue.get_parent()
        chain.detach_child(residue.id)

    return structure


def save_structure(structure, output_file_path):
    """
    Save structure to a file.
    """
    if output_file_path.endswith(".pdb"):
        io = PDBIO()
    elif output_file_path.endswith(".cif"):
        io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file_path)


def to_bio(input_file_path):
    if input_file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif input_file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", input_file_path)
    return structure

def main(argv):
    input_file_path = flags.FLAGS.input_file_path
    output_file_path = flags.FLAGS.output_file_path
    threshold = flags.FLAGS.threshold
    hb_allowance = flags.FLAGS.hb_allowance
    plddt_threshold = flags.FLAGS.plddt_threshold

    structure = to_bio(input_file_path)
    structure = remove_clashes(structure, threshold, hb_allowance)
    structure = remove_low_plddt(structure, plddt_threshold)

    if output_file_path:
        save_structure(structure, output_file_path)


if __name__ == '__main__':
    flags.DEFINE_string("input_file_path", None, "Path to the input PDB or CIF file")
    flags.DEFINE_string("output_file_path", None, "Path to save the output file. Optional if --save is False")
    flags.DEFINE_float("threshold", 0.9, "Threshold for VDW overlap to identify clashes")
    flags.DEFINE_float("hb_allowance", 0.4, "Allowance for hydrogen bonding (default: 0.0)")
    flags.DEFINE_float("plddt_threshold", 50, "Threshold for pLDDT score (default: 50)")
    flags.mark_flags_as_required(["input_file_path"])
    app.run(main)