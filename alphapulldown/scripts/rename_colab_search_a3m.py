#!/usr/bin/env python3
import glob
import os
import sys
from itertools import groupby


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


def fastafn2seqs(fasta_fn):
    """Extract sequences from a FASTA file."""
    seqs = []
    with open(fasta_fn) as f:
        for header_str, seq in fasta_iter(f):
            seqs.append((header_str, seq))
    return seqs


def main(input_fasta_fn=None):
    """Rename all .a3m files in the current directory.

    Old colabfold: Rename all .a3m files to the name of the first sequence in the respective file.
    New colabfold: Rename all .a3m files to the names used in the input fasta file.
    """
    fn_idx = 0
    files = sorted(glob.glob("*.a3m"), key=lambda x: int(os.path.splitext(x)[0]))

    if input_fasta_fn:
        in_seqs = fastafn2seqs(input_fasta_fn)

    for file in files:
        this_seqs = fastafn2seqs(file)
        name = this_seqs[0][0]

        if name == '101':  # New colabfold
            if not input_fasta_fn:
                raise ValueError('Please provide the input FASTA file used for colabfold_search')
            name = in_seqs[fn_idx][0]
            this_seqs[0] = (name, this_seqs[0][1])

        outfile = f"{name}.a3m"
        print(f"Renaming {file} to {outfile}")
        with open(outfile, 'w') as f:
            for header, seq in this_seqs:
                f.write(f">{header}\n{seq}\n")
        os.remove(file)

        fn_idx += 1


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()