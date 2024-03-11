#!python3
""" CLI inferface for performing structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Dict
import argparse
from os import makedirs
from os.path import exists, join

from alphapulldown.run_multimer_jobs import create_custom_info
from alphapulldown.utils import create_interactors
from alphapulldown.objects import MultimericObject
from alphapulldown.folding_backend import backend


def parse_args():
    parser = argparse.ArgumentParser(description="Run protein folding.")
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        required=True,
        help="Folds in format [fasta_path:number:start-stop],[...],.",
    ),
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        type=str,
        required=True,
        help="Path to output directory. Will be created if not exists.",
    ),
    parser.add_argument(
        "--num_cycle",
        dest="num_cycle",
        type=int,
        required=False,
        default=3,
        help="Number of recycles, defaults to 3.",
    ),
    parser.add_argument(
        "--num_predictions_per_model",
        dest="num_predictions_per_model",
        type=int,
        required=False,
        default=1,
        help="Number of predictions per model, defaults to 1.",
    ),
    parser.add_argument(
        "--data_directory",
        dest="data_directory",
        type=str,
        required=True,
        help="Path to directory containing model weights and parameters.",
    ),
    parser.add_argument(
        "--features_directory",
        dest="features_directory",
        type=str,
        nargs="+",
        required=True,
        help="Path to computed monomer features.",
    ),
    parser.add_argument(
        "--no_pair_msa",
        dest="no_pair_msa",
        action="store_true",
        default=False,
        help="Do not pair the MSAs when constructing multimer objects.",
    ),
    parser.add_argument(
        "--gradient_msa_depth",
        dest="gradient_msa_depth",
        action="store_true",
        default=None,
        help="Run predictions for each model with logarithmically distributed MSA depth.",
    ),
    parser.add_argument(
        "--multimeric_template",
        dest="multimeric_template",
        action="store_true",
        default=None,
        help="Whether to use multimeric templates.",
    ),
    parser.add_argument(
        "--model_names",
        dest="model_names",
        type=str,
        default=None,
        help="Names of models to use, e.g. model_2_multimer_v3 (default: all models).",
    ),
    parser.add_argument(
        "--msa_depth",
        dest="msa_depth",
        type=int,
        default=None,
        help="Number of sequences to use from the MSA (by default is taken from AF model config).",
    ),
    parser.add_argument(
        "--protein_delimiter",
        dest="protein_delimiter",
        type=str,
        default="_",
        help="Delimiter for proteins of a single fold.",
    ),
    parser.add_argument(
        "--fold_backend",
        dest="fold_backend",
        type=str,
        default="alphafold",
        choices=list(backend._BACKEND_REGISTRY.keys()),
        help="Folding backend that should be used for structure prediction.",
    ),
    parser.add_argument(
        "--crosslinks",
        dest="crosslinks",
        type=str,
        default=None,
        required=False,
        help="Path to crosslink information pickle for AlphaLink.",
    ),
    parser.add_argument(
        "--description_file",
        dest="description_file",
        type=str,
        default=None,
        required=False,
        help="Path to the text file with multimeric template instruction.",
    ),
    parser.add_argument(
        "--path_to_mmt",
        dest="path_to_mmt",
        type=str,
        default=None,
        required=False,
        help="Path to directory with multimeric template mmCIF files.",
    ),
    parser.add_argument(
        "--compress_result_pickles",
        dest="compress_result_pickles",
        action="store_true",
        required=False,
        help="Whether the result pickles are going to be gzipped. Default False.",
    ),
    parser.add_argument(
        "--remove_result_pickles",
        dest="remove_result_pickles",
        action="store_true",
        required=False,
        help="Whether the result pickles that do not belong to the best"
        "model are going to be removed. Default is False.",
    ),
    args = parser.parse_args()

    makedirs(args.output_directory, exist_ok=True)

    formatted_folds, missing_features, unique_features = [], [], []
    protein_folds = [x.split(":") for x in args.input.split(args.protein_delimiter)]
    for protein_fold in protein_folds:
        name, number, region = None, 1, "all"

        match len(protein_fold):
            case 1:
                name = protein_fold[0]
            case 2:
                name, number = protein_fold[0], protein_fold[1]
                if ("-") in protein_fold[1]:
                    number = 1
                    region = protein_fold[1].split("-")
            case 3:
                name, number, region = protein_fold

        number = int(number)
        if len(region) != 2 and region != "all":
            raise ValueError(f"Region {region} is malformatted expected start-stop.")

        if len(region) == 2:
            region = [tuple(int(x) for x in region)]

        unique_features.append(name)
        for monomer_dir in args.features_directory:
            if exists(join(monomer_dir, f"{name}.pkl")):
                continue
            missing_features.append(name)

        formatted_folds.extend([{name: region} for _ in range(number)])

    missing_features = set(missing_features)
    if len(missing_features):
        raise FileNotFoundError(
            f"{missing_features} not found in {args.features_directory}"
        )

    args.parsed_input = formatted_folds

    return args


def predict_structure(
    multimeric_object: MultimericObject,
    output_dir: str,
    model_flags: Dict,
    postprocess_flags: Dict,
    random_seed: int = 42,
    fold_backend: str = "alphafold",
) -> None:
    """
    Predict structural features of multimers using specified models and configurations.

    Parameters
    ----------
    multimer : MultimericObject
        An instance of `MultimericObject` representing the multimeric structure(s) for which
        predictions are to be made. These objects should be created using functions like
        `create_multimer_objects()`, `create_custom_jobs()`, or `create_homooligomers()`.
    output_directory : str
        The directory path where the prediction results will be saved.
    model_flags : Dict
        Dictionary of flags passed to the respective backend's setup function.
    model_flags : Dict
        Dictionary of flags passed to the respective backend's postprocess function.
    random_seed : int, optional
        The random seed for initializing the prediction process to ensure reproducibility.
        Default is 42.
    fold_backend : str, optional
        Backend used for folding, defaults to alphafold.
    """
    backend.change_backend(backend_name=fold_backend)

    model_runners = backend.setup(**model_flags, multimeric_object=multimeric_object)

    backend.predict(
        **model_runners,
        multimeric_object=multimeric_object,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    backend.postprocess(
        **postprocess_flags,
        multimeric_object=multimeric_object,
        output_dir=output_dir,
    )


def main():
    args = parse_args()

    data = create_custom_info(args.parsed_input)
    interactors = create_interactors(data, args.features_directory, 0)
    multimer = interactors[0]
    if len(interactors) > 1:
        multimer = MultimericObject(
            interactors=interactors,
            pair_msa=not args.no_pair_msa,
            multimeric_mode=args.multimeric_template,
            multimeric_template_meta_data=args.description_file,
            multimeric_template_dir=args.path_to_mmt,
        )

    if not isinstance(multimer, MultimericObject):
        multimer.input_seqs = [multimer.sequence]

    # TODO: Add backend specific flags here
    flags_dict = {
        "model_name": "monomer_ptm",
        "num_cycle": args.num_cycle,
        "model_dir": args.data_directory,
        "num_multimer_predictions_per_model": args.num_predictions_per_model,
        "multimeric_object": multimer,
        "crosslinks": args.crosslinks,
    }

    if isinstance(multimer, MultimericObject):
        flags_dict["model_name"] = "multimer"
        flags_dict["gradient_msa_depth"] = (args.gradient_msa_depth,)
        flags_dict["model_names_custom"] = args.model_names
        flags_dict["msa_depth"] = args.msa_depth

    postprocess_flags = {
        "zip_pickles": args.compress_result_pickles,
        "remove_pickles": args.remove_result_pickles,
    }

    predict_structure(
        multimeric_object=multimer,
        output_directory=args.output_directory,
        model_flags=flags_dict,
        fold_backend=args.fold_backend,
        postprocess_flags=postprocess_flags,
    )


if __name__ == "__main__":
    main()