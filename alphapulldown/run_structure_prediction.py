#!python3
""" CLI inferface for performing structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
from os import makedirs
from os.path import exists, join

from alphapulldown.run_multimer_jobs import create_custom_info
from alphapulldown.utils import create_model_runners_and_random_seed, create_interactors
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
        help="Path to data directory.",
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
        default=";",
        help="Delimiter for proteins of a singel fold.",
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


def predict_multimer(
    multimer: MultimericObject,
    num_recycles: int,
    data_directory: str,
    num_predictions_per_model: int,
    output_directory: str,
    gradient_msa_depth: bool = False,
    model_names: str = None,
    msa_depth: int = None,
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
    num_recycles : int
        The number of recycles to be used during the prediction process.
    data_directory : str
        The directory path where input data for the prediction process is stored.
    num_predictions_per_model : int
        The number of predictions to generate per model.
    output_directory : str
        The directory path where the prediction results will be saved.
    gradient_msa_depth : bool, optional
        A flag indicating whether to adjust the MSA depth based on gradients. Default is False.
    model_names : str, optional
        The names of the models to be used for prediction. If not provided, a default set of
        models is used. Default is None.
    msa_depth : int, optional
        Specifies the depth of the MSA (Multiple Sequence Alignment) to be used. If not
        provided, a default value based on the model configuration is used. Default is None.
    random_seed : int, optional
        The random seed for initializing the prediction process to ensure reproducibility.
        Default is 42.
    fold_backend : str, optional
        Backend used for folding, defaults to alphafold.
    """

    flags_dict = {
        "model_preset": "monomer_ptm",
        "random_seed": random_seed,
        "num_cycle": num_recycles,
        "data_dir": data_directory,
        "num_multimer_predictions_per_model": num_predictions_per_model,
    }

    if isinstance(multimer, MultimericObject):
        flags_dict["model_preset"] = "multimer"
        flags_dict["gradient_msa_depth"] = gradient_msa_depth
        flags_dict["model_names_custom"] = model_names
        flags_dict["msa_depth"] = msa_depth
    else:
        multimer.input_seqs = [multimer.sequence]

    model_runners, random_seed = create_model_runners_and_random_seed(**flags_dict)

    backend.change_backend(backend_name=fold_backend)

    backend.predict(
        model_runners=model_runners,
        output_dir=output_directory,
        feature_dict=multimer.feature_dict,
        random_seed=random_seed,
        fasta_name=multimer.description,
        seqs=multimer.input_seqs,
    )
    backend.postprocess(
        multimer=multimer,
        output_path=output_directory,
        zip_pickles=False,
        remove_pickles=False,
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
        )

    predict_multimer(
        multimer=multimer,
        num_recycles=args.num_cycle,
        data_directory=args.data_directory,
        num_predictions_per_model=args.num_predictions_per_model,
        output_directory=args.output_directory,
        gradient_msa_depth=args.gradient_msa_depth,
        model_names=args.model_names,
        msa_depth=args.msa_depth,
    )


if __name__ == "__main__":
    main()
