#!/usr/bin/env python3
import sys
from itertools import groupby
import re

"""
Rename Uniprot names in FASTA file to uniprot IDs
(split by | and take second element)
"""


def fasta_iter(fh):
    """Return iterator over FASTA file with multiple sequences.

    Modified from Brent Pedersen
    Correct Way To Parse A Fasta File In Python
    given a fasta file. yield tuples of header, sequence

    :param fh: File Handle to the FASTA file

    :return: 2-element tuple with header and sequence strings
    """

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)

out_lines = []

with open(sys.argv[1]) as f:
    for headerStr, seq in fasta_iter(f):
        items = re.split('[ \|]', headerStr)
        out_lines.append(f'>{items[1]}')
        out_lines.append(seq)
    f.close()
print("\n".join(out_lines))