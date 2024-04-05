#
# Author: Grzegorz Chojnowski @ EMBL-Hamburg
#

import os, sys, re
import glob
import iotbx
import cctbx
import iotbx.pdb
import re
import py3Dmol
from scitbx import matrix
from scitbx.math import superpose
import numpy as np

PLDDT_BANDS = [(50, '#FF7D45'), (70, '#FFDB13'), (90, '#65CBF3'), (100, '#0053D6')]

def parse_pdbstring(pdb_string):

    # there may be issues with repeated BREAK lines, that we do not use here anyway
    pdb_string_lines = []
    for _line in pdb_string.splitlines():
        if re.match(r"^BREAK$", _line):
            continue
        pdb_string_lines.append(_line)

    # arghhhh, why do you guys keep changing the interface?
    inp = iotbx.pdb.input(source_info=None, lines=pdb_string_lines)
    try:
        return inp.construct_hierarchy(sort_atoms=False), inp.crystal_symmetry()
    except:
        return inp.construct_hierarchy(), inp.crystal_symmetry()


# -------------------------------------------------------------------------


def read_ph(ifname, selstr=None, verbose=True):

    if verbose:
        print(" ==> Parsing a PDB/mmCIF file: %s" % ifname)

    with open(ifname, "r") as ifile:
        ph, symm = parse_pdbstring(ifile.read())

    if selstr is None:
        return ph, symm

    sel_cache = ph.atom_selection_cache()
    isel = sel_cache.iselection

    phsel = ph.select(isel(selstr))

    return phsel, symm

def parse_results(output, color=None, models=5, multimer=False):

    if color is None:
        color = ["lDDT", "rainbow", "chain"][0]

    datadir = os.path.expanduser(output)

    # now sorted numerically
    pattern = r"ranked_(\d+)\.pdb"
    pdb_fnames = sorted(glob.glob("%s/ranked*.pdb" % datadir),
                        key=lambda x: int(re.search(pattern,x).group(1)))

    ph_array = []
    for idx, fn in enumerate(pdb_fnames[:models]):
        _ph, _symm = read_ph(fn)
        if len(ph_array) > 0:
            _s = superpose.least_squares_fit(
                ph_array[0].atoms().extract_xyz(),
                _ph.atoms().extract_xyz(),
                method=["kearsley", "kabsch"][0],
            )
            rtmx = matrix.rt((_s.r, _s.t))
            _ph.atoms().set_xyz(new_xyz=rtmx * _ph.atoms().extract_xyz())

        ph_array.append(_ph)

    chain_ids = [_.id for _ in _ph.chains()]

    if len(chain_ids) > 1 and color == "chain":
        view = py3Dmol.view(
            width=900,
            height=600,
            viewergrid=(1, 2),
        )

        viewer = (0, 0)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

        viewer = (0, 1)

        set_b_to_plddtbands(ph_array[0])

        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="lDDT"
        )

    else:
        frames = min(models, len(ph_array))
        view = py3Dmol.view(
            width=400 * frames,
            height=400,
            viewergrid=(1, frames),
        )

        for idx, _ph in enumerate(ph_array):
            viewer = (0, idx)

            if color=="lDDT": set_b_to_plddtbands(_ph)

            view.addModel(_ph.as_pdb_string(), "pdb", viewer=viewer)
            view.zoomTo(viewer=viewer)

            set_3dmol_styles(view, viewer, chain_ids=chain_ids, color=color)

    view.show()


def parse_results_colour_chains(output, color=None, models=5, multimer=False):

    if color is None:
        color = ["chain"][0]

    datadir = os.path.expanduser(output)
    
    # now sorted numerically
    pattern = r"ranked_(\d+)\.pdb"
    pdb_fnames = sorted(glob.glob("%s/ranked*.pdb" % datadir),
                        key=lambda x: int(re.search(pattern,x).group(1)))

    ph_array = []
    for idx, fn in enumerate(pdb_fnames[:models]):
        _ph, _symm = read_ph(fn)
        if len(ph_array) > 0:
            _s = superpose.least_squares_fit(
                ph_array[0].atoms().extract_xyz(),
                _ph.atoms().extract_xyz(),
                method=["kearsley", "kabsch"][0],
            )
            rtmx = matrix.rt((_s.r, _s.t))
            _ph.atoms().set_xyz(new_xyz=rtmx * _ph.atoms().extract_xyz())

        ph_array.append(_ph)

    chain_ids = [_.id for _ in _ph.chains()]

    if len(chain_ids) > 1 and color == None:
        view = py3Dmol.view(
            width=900,
            height=600,
            viewergrid=(1, 2),
        )

        viewer = (0, 0)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

        viewer = (0, 1)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

    else:
        frames = min(models, len(ph_array))
        view = py3Dmol.view(
            width=400 * frames,
            height=400,
            viewergrid=(1, frames),
        )

        for idx, _ph in enumerate(ph_array):
            viewer = (0, idx)

            if color=="lDDT": set_b_to_plddtbands(_ph)

            view.addModel(_ph.as_pdb_string(), "pdb", viewer=viewer)
            view.zoomTo(viewer=viewer)

            set_3dmol_styles(view, viewer, chain_ids=chain_ids, color=color)

    view.show()


# ------------------------------------------------------


def set_b_to_plddtbands(ph):

    plddt_lims = np.array([_[0] for _ in PLDDT_BANDS])
    for resi in ph.residue_groups():
        resi.atoms().set_b(new_b=cctbx.array_family.flex.double(resi.atoms().size(), float(np.argmax(plddt_lims>resi.atoms()[0].b))))

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
