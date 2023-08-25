from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from collections import defaultdict
import logging

DONORS_ACCEPTORS = ['N', 'O', 'S']
VDW_RADII = defaultdict(lambda: 1.5)
VDW_RADII.update({'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8, 'CL': 1.75})

def is_potential_hbond(atom1, atom2):
    if (atom1.element in DONORS_ACCEPTORS and atom2.element in DONORS_ACCEPTORS) or \
            (atom2.element in DONORS_ACCEPTORS and atom1.element in DONORS_ACCEPTORS):
        return True
    return False

def remove_clashes(structure, threshold=0.9, hb_allowance=0.4):
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
    for residue in clashing_residues:
        chain = residue.get_parent()
        chain.detach_child(residue.id)

    return len(list(model.get_residues())), len(clashing_residues), len(clashing_atoms)

def remove_low_plddt(structure, plddt_threshold=50):
    model = structure[0]
    low_plddt_residues = set()
    for residue in model.get_residues():
        if any(atom.get_bfactor() < plddt_threshold for atom in residue):
            low_plddt_residues.add(residue)

    for residue in low_plddt_residues:
        chain = residue.get_parent()
        chain.detach_child(residue.id)

    return len(list(model.get_residues())), len(low_plddt_residues)
