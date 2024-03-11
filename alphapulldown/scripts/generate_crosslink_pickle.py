#!/usr/bin/env python3

# Adapted from https://github.com/Rappsilber-Laboratory/AlphaLink2/blob/main/generate_crosslink_pickle.py
# A script to cross-link input pickles 
# #

import numpy as np
from argparse import ArgumentParser
import pickle
import gzip


def parse_arguments():
    parser = ArgumentParser(description='Generate crosslink pickle')
    parser.add_argument('--csv',
                        help='CSV with contacts: i chain1 j chain2 FDR',
                        required=True)
    parser.add_argument('--output',
                        help='Output pickle',
                        required=True)                   
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    links = np.loadtxt(args.csv,dtype=str)

    if len(links.shape) == 1:
        links = np.array([links])

    crosslinks = {}

    for i,chain1,j,chain2,fdr in links:
        i = int(i)
        j = int(j)
        fdr = float(fdr)
        if not chain1 in crosslinks:
            crosslinks[chain1] = {}
        if not chain2 in crosslinks[chain1]:
            crosslinks[chain1][chain2] = []

        crosslinks[chain1][chain2].append((i-1,j-1,fdr))

    pickle.dump(crosslinks, gzip.open(args.output,'wb'))

if __name__ == "__main__":
    main()
