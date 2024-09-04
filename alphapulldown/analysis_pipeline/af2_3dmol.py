#
# Author: Grzegorz Chojnowski @ EMBL-Hamburg
#

import os
import glob
from Bio.PDB import PDBIO, PDBParser, Superimposer
from io import StringIO  
import re
import py3Dmol
import numpy as np

PLDDT_BANDS = [(50, '#FF7D45'), (70, '#FFDB13'), (90, '#65CBF3'), (100, '#0053D6')]
COLOR_SCHEMES = "lDDT,rainbow,chain".split(',')

# -------------------------------------------------------------------------

def get_CAs(structure):
    CA_atoms = []
    for _chain in structure :
        for _res in _chain:
            CA_atoms.append(_res['CA'])                
            
    return CA_atoms

# ------------------------------------------------------

def bio2pdbstring(structure):
    io=PDBIO()
    io.set_structure(structure)
    fo = StringIO()
    io.save(fo)
    fo.seek(0)
    return fo.read()

# ------------------------------------------------------

def parse_results(output, color=None, models=5, multimer=False):

    if color not in COLOR_SCHEMES:
        color = COLOR_SCHEMES[0]

    datadir = os.path.expanduser(output)

    # now sorted numerically
    pattern = r"ranked_(\d+)\.pdb"
    pdb_fnames = sorted(glob.glob("%s/ranked*.pdb" % datadir),
                        key=lambda x: int(re.search(pattern,x).group(1)))

    parser = PDBParser()
    sup = Superimposer()

    bio_array = []
    
    for idx, fn in enumerate(pdb_fnames[:models]):
        structure = parser.get_structure("AF", fn)[0]
        
        if len(bio_array) > 0:
            
            sup.set_atoms(get_CAs(bio_array[0]), get_CAs(structure))
            sup.apply(structure)
            

        bio_array.append(structure)

    chain_ids = [_.id for _ in bio_array[0]]    

    if len(chain_ids) > 1 and color == "chain":
        view = py3Dmol.view(
            width=900,
            height=600,
            viewergrid=(1, 2),
        )

        viewer = (0, 0)
        view.addModel(bio2pdbstring(bio_array[0]), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=chain_ids, color="chain"
        )

        viewer = (0, 1)

        set_b_to_plddtbands_bio(bio_array[0])

        view.addModel(bio2pdbstring(bio_array[0]), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=chain_ids, color="lDDT"
        )

    else:
        frames = min(models, len(bio_array))
        view = py3Dmol.view(
            width=400 * frames,
            height=400,
            viewergrid=(1, frames),
        )

        for idx, _ph in enumerate(bio_array):
            viewer = (0, idx)

            if color=="lDDT": set_b_to_plddtbands_bio(_ph)

            view.addModel(bio2pdbstring(_ph), "pdb", viewer=viewer)

            
            view.zoomTo(viewer=viewer)

            set_3dmol_styles(view, viewer, chain_ids=chain_ids, color=color)

    view.show()

# ------------------------------------------------------

def parse_results_colour_chains(output, color=None, models=5, multimer=False):
    # for backward compatibility ONLY
    parse_results(output, color='chain', models=models, multimer=multimer)


# ------------------------------------------------------

def set_b_to_plddtbands_bio(structure):

    plddt_lims = np.array([_[0] for _ in PLDDT_BANDS])
    
    for chain in structure:
        for resi in chain:
            for atm in resi:
                atm.set_bfactor( float(np.argmax(plddt_lims>atm.get_bfactor())) )
            

# ------------------------------------------------------

def set_3dmol_styles(
    view,
    viewer,
    chain_ids=1,
    color=["lDDT", "rainbow", "chain"][0],
    show_sidechains=False,
    show_mainchains=False,
):

    """
    borrowed from colabfolds notebook at
    https://github.com/sokrypton/ColabFold/blob/main/colabfold/pdb.py
    """

    if color == "lDDT":

        color_map = {i: band[1] for i, band in enumerate(PLDDT_BANDS)}

        view.setStyle(
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        'map': color_map
                    }
                }
            },
            viewer=viewer,
        )
    elif color == "rainbow":
        view.setStyle({"cartoon": {"color": "spectrum"}}, viewer=viewer)

    elif color == "chain":
        for cid, color in zip(
            chain_ids,
            [
                "lime",
                "cyan",
                "magenta",
                "yellow",
                "salmon",
                "white",
                "blue",
                "orange",
                "black",
                "green",
                "gray",
            ]
            * 2,
        ):
            view.setStyle({"chain": cid}, {"cartoon": {"color": color}}, viewer=viewer)
    if show_sidechains:
        BB = ["C", "O", "N"]
        view.addStyle(
            {
                "and": [
                    {"resn": ["GLY", "PRO"], "invert": True},
                    {"atom": BB, "invert": True},
                ]
            },
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
            {"sphere": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "PRO"}, {"atom": ["C", "O"], "invert": True}]},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
    if show_mainchains:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
            viewer=viewer,
        )
