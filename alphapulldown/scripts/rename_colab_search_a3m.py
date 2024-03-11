#!/usr/bin/env python3
import glob, os
from itertools import groupby

"""
Rename all .a3m files in the current directory
to the name of the first sequence in the respective file.
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

def get_first_seq_name(fasta_fn):
    with open(fasta_fn) as f:
        for headerStr, seq in fasta_iter(f):
            return headerStr

for file in glob.glob("*.a3m"):
    name = get_first_seq_name(file)
    outfile = name+'.a3m'
    print(f'Renaming {file} to {outfile}')
    os.rename(file, outfile)
