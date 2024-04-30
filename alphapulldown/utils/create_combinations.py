""" Computes cartesian product of lines in multiple files.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import argparse
import itertools
from contextlib import nullcontext
from typing import List, Union, TextIO

def read_file(filepath : str):
    with open(filepath, mode = "r", encoding = "utf-8") as file:
        lines = file.read().splitlines()
    return list(line.lstrip().rstrip() for line in lines if line)

def process_files(input_files : List[str], 
                  output_path : Union[str, TextIO] = None, 
                  delimiter : str = '+',
                  exclude_permutations : bool = True
                  ):
    """Process the input files to compute the Cartesian product and write to the output file."""
    lists_of_lines = [read_file(filepath) for filepath in input_files]
    cartesian_product = list(itertools.product(*lists_of_lines))

    if exclude_permutations:
        keep = []
        unique_elements = set()
        for combination in cartesian_product:
            sorted_combination = tuple(sorted(combination))
            if sorted_combination in unique_elements:
                continue
            unique_elements.add(sorted_combination)
            keep.append(combination)
        cartesian_product = keep

    if output_path is None:
        return cartesian_product
    else:
        context_manager = nullcontext(output_path)
        if isinstance(output_path, str):
            context_manager = open(output_path, mode = "w", encoding = "utf-8")
        
        with context_manager as output_file:
            for combination in cartesian_product:
                output_file.write(delimiter.join(combination) + '\n')

def main():
    parser = argparse.ArgumentParser(
        description="Compute cartesian product of lines in multiple files."
    )
    parser.add_argument('input_files', nargs='+', help="List of input files.")
    parser.add_argument('--output', required=True, help="Path to output file.")
    parser.add_argument('--delimiter', default='_', help="Delimiter for line from each file.")

    args = parser.parse_args()

    process_files(args.input_files, args.output, args.delimiter)

if __name__ == "__main__":
    main()
