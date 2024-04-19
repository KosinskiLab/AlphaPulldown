""" Implements parsing logic for Snakemake pipeline.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from os import makedirs, symlink
from os.path import exists, splitext, basename, join
from typing import Tuple, List, Set, Dict


class InputParser:
    def __init__(
        self,
        fold_specifications: Tuple[str],
        sequences_by_origin: Dict[str, List[str]],
        sequences_by_fold: Dict[str, Set],
    ):
        self.fold_specifications = fold_specifications
        self.sequences_by_origin = sequences_by_origin
        self.sequences_by_fold = sequences_by_fold

        unique_sequences = set()
        for value in self.sequences_by_origin.values():
            unique_sequences.update(
                set([splitext(basename(x))[0] for x in value])
            )
        self.unique_sequences = unique_sequences

    @staticmethod
    def _strip_path_and_extension(filepath : str) -> str:
        return splitext(basename(filepath))[0]

    @staticmethod
    def _parse_alphaabriss_format(
        fold_specifications: List[str],
        protein_delimiter : str = "_"
    ) -> Tuple[Dict[str, List[str]], Dict[str, Set]]:
        unique_sequences, sequences_by_fold = set(), {}

        for fold_specification in fold_specifications:
            sequences = set()
            clean_fold_specification = []
            for fold in fold_specification.split(protein_delimiter):
                fold = fold.split(":")
                sequences.add(fold[0])

                protein_name = splitext(basename(fold[0]))[0]
                clean_fold_specification.append(":".join([protein_name, *fold[1:]]))

            clean_fold_specification = protein_delimiter.join([str(x) for x in clean_fold_specification])

            unique_sequences.update(sequences)
            sequences_by_fold[clean_fold_specification] = {splitext(basename(x))[0] for x in sequences}

        sequences_by_origin = {
            "uniprot" : [],
            "local" : []
        }
        for sequence in unique_sequences:
            if not exists(sequence):
                sequences_by_origin["uniprot"].append(sequence)
                continue
            sequences_by_origin["local"].append(sequence)

        return sequences_by_origin, sequences_by_fold

    def symlink_local_files(self, output_directory : str) -> None:
        makedirs(output_directory, exist_ok = True)
        for file in self.sequences_by_origin["local"]:
            symlink(file, join(output_directory, basename(file)))
        return None

    @classmethod
    def from_file(cls, filepath: str, file_format: str = "alphaabriss", protein_delimiter : str = "_"):
        with open(filepath, mode="r") as infile:
            data = [line.strip() for line in infile.readlines() if len(line.strip())]
            data = tuple(set(data))

        match file_format:
            case "alphaabriss":
                ret = cls._parse_alphaabriss_format(
                    fold_specifications = data, protein_delimiter=protein_delimiter
                )
                sequences_by_origin, sequences_by_fold = ret

            case _:
                raise ValueError(f"Format {file_format} is not supported.")

        fold_specifications = list(sequences_by_fold.keys())
        return cls(
            fold_specifications=fold_specifications,
            sequences_by_origin=sequences_by_origin,
            sequences_by_fold=sequences_by_fold,
        )
