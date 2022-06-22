#
# Author: Grzegorz Chojnowski @ EMBL-Hamburg
#

import iotbx
import re
import iotbx.pdb


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
