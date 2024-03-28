#!python3
""" CLI inferface for performing structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import argparse
from os import makedirs
from typing import Dict, List, Union, Tuple
from os.path import exists, join
from absl import logging
from alphapulldown.folding_backend import backend
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.modelling_setup import create_interactors, parse_fold,create_custom_info

logging.set_verbosity(logging.INFO)

def add_required_args(parser : argparse.ArgumentParser) -> None :
    """
    Parse arguments are are required
    """
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        nargs="+",
        required=True,
        help="Folds in format [fasta_path:number:start-stop],[...],.",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        type=str,
        required=True,
        help="Path to output directory. Will be created if not exists.",
    )
    parser.add_argument(
        "--data_directory",
        dest="data_directory",
        type=str,
        required=True,
        help="Path to directory containing model weights and parameters.",
    )
    parser.add_argument(
        "--features_directory",
        dest="features_directory",
        type=str,
        nargs="+",
        required=True,
        help="Path to computed monomer features.",
    )

def add_alphafold_settings(parser : argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num_cycle",
        dest="num_cycle",
        type=int,
        required=False,
        default=3,
        help="Number of recycles, defaults to 3.",
    )
    parser.add_argument(
        "--num_predictions_per_model",
        dest="num_predictions_per_model",
        type=int,
        required=False,
        default=1,
        help="Number of predictions per model, defaults to 1.",
    )
    parser.add_argument(
        "--no_pair_msa",
        dest="no_pair_msa",
        action="store_true",
        default=False,
        help="Do not pair the MSAs when constructing multimer objects.",
    )
    parser.add_argument(
        "--gradient_msa_depth",
        dest="gradient_msa_depth",
        action="store_true",
        default=None,
        help="Run predictions for each model with logarithmically distributed MSA depth.",
    )
    parser.add_argument(
        "--multimeric_template",
        dest="multimeric_template",
        action="store_true",
        default=None,
        help="Whether to use multimeric templates.",
    )
    parser.add_argument(
        "--model_names",
        dest="model_names",
        type=str,
        default=None,
        help="Names of models to use, e.g. model_2_multimer_v3 (default: all models).",
    )
    parser.add_argument(
        "--msa_depth",
        dest="msa_depth",
        type=int,
        default=None,
        help="Number of sequences to use from the MSA (by default is taken from AF model config).",
    )
    parser.add_argument(
        "--description_file",
        dest="description_file",
        type=str,
        default=None,
        required=False,
        help="Path to the text file with multimeric template instruction.",
    )
    parser.add_argument(
        "--path_to_mmt",
        dest="path_to_mmt",
        type=str,
        default=None,
        required=False,
        help="Path to directory with multimeric template mmCIF files.",
    )
    parser.add_argument(
        "--desired_num_res",
        dest="desired_num_res",
        type=int,
        required=False,
        default=None,
        help="A desired number of residues to pad"
    )
    parser.add_argument(
        "--desired_num_msa",
        dest="desired_num_msa",
        type=int,
        required=False,
        default=None,
        help="A desired number of msa to pad"
    )

def add_alphalink2_settings(parser : argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--crosslinks",
        dest="crosslinks",
        type=str,
        default=None,
        required=False,
        help="Path to crosslink information pickle for AlphaLink.",
    )

def add_post_processing_settgins(parser : argparse.ArgumentParser) -> None:
    parser.add_argument(
            "--compress_result_pickles",
            dest="compress_result_pickles",
            action="store_true",
            required=False,
            help="Whether the result pickles are going to be gzipped. Default False.",
        )
    parser.add_argument(
        "--remove_result_pickles",
        dest="remove_result_pickles",
        action="store_true",
        required=False,
        help="Whether the result pickles that do not belong to the best"
        "model are going to be removed. Default is False.",
    )
    parser.add_argument(
        "--use_ap_style",
        dest="use_ap_style",
        action="store_true",
        required=False,
        help="Change output directory to include a description of the fold as seen "
        "in previous alphapulldown versions.",
    )
    parser.add_argument(
        "--use_gpu_relax",
        dest="use_gpu_relax",
        action="store_true",
        required=False,
        help="Whether to run Amber relaxation on GPU. Default is True"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run protein folding.")
    parser.add_argument(
        "--protein_delimiter",
        dest="protein_delimiter",
        type=str,
        default="+",
        help="Delimiter for proteins of a single fold.",
    ),
    parser.add_argument(
        "--fold_backend",
        dest="fold_backend",
        type=str,
        default="alphafold",
        choices=list(backend._BACKEND_REGISTRY.keys()),
        help="Folding backend that should be used for structure prediction.",
    )
    add_required_args(parser)
    add_alphafold_settings(parser)
    add_alphalink2_settings(parser)
    add_post_processing_settgins(parser)
    args = parser.parse_args()
    return parse_fold(args)


def predict_structure(
    objects_to_model: List[Dict[Union[MultimericObject, MonomericObject, ChoppedObject], str]],
    model_flags: Dict,
    postprocess_flags: Dict,
    random_seed: int = 42,
    fold_backend: str = "alphafold"
) -> None:
    """
    Predict structural features of multimers using specified models and configurations.

    Parameters
    ----------
    objects_to_model : A list of dictionareis. Each dicionary has a key of MultimericObject or MonomericObject or ChoppedObject
       which is an instance of `MultimericObject` representing the multimeric/monomeric structure(s).
       for which predictions are to be made. These objects should be created using functions like
    `create_multimer_objects()`, `create_custom_jobs()`, or `create_homooligomers()`.
    The value of each dictionary is the corresponding output_dir to save the modelling results. 
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
    model_runners_and_configs = backend.setup(**model_flags)

    backend.predict(
        **model_runners_and_configs,
        multimeric_object=objects_to_model,
        random_seed=random_seed,
        **model_flags
    )
    backend.postprocess(
        **postprocess_flags,
        multimeric_object=objects_to_model
    )

def pre_modelling_setup(interactors : List[Union[MonomericObject, ChoppedObject]], 
                        args) -> Tuple[Union[MultimericObject,
                                            MonomericObject, 
                                            ChoppedObject], dict, dict, str]:
    """
    A function that sets up objects that to be modelled 
    and settings dictionaries 

    Args:
    inteactors: A list of MOnomericobejct or ChoppedObject. If len(interactors) ==1, 
    that means a monomeric modelling job should be done. Otherwise, it will be a multimeric modelling job
    args: argparse results

    Return:
    A MultimericObject or MonomericObject
    A dictionary of flags_dict
    A dicionatry of postprocessing_flags
    output_directory of this particular modelling job
    """
    if len(interactors) > 1:
        # this means it's going to be a MultimericObject
        object_to_model = MultimericObject(
            interactors=interactors,
            pair_msa=not args.no_pair_msa,
            multimeric_mode=args.multimeric_template,
            multimeric_template_meta_data=args.description_file,
            multimeric_template_dir=args.path_to_mmt,
        )
    else:
        # means it's going to be a MonomericObject or a ChoppedObject
        object_to_model= interactors[0]
        object_to_model.input_seqs = [object_to_model.sequence]

    # TODO: Add backend specific flags here
    flags_dict = {
        "model_name": "monomer_ptm",
        "num_cycle": args.num_cycle,
        "model_dir": args.data_directory,
        "num_multimer_predictions_per_model": args.num_predictions_per_model,
        "crosslinks": args.crosslinks,
        "use_gpu_relax": args.use_gpu_relax,
        "desired_num_res": args.desired_num_res,
        "desired_num_msa": args.desired_num_msa
    }

    if isinstance(object_to_model, MultimericObject):
        flags_dict["model_name"] = "multimer"
        flags_dict["gradient_msa_depth"] = (args.gradient_msa_depth,)
        flags_dict["model_names_custom"] = args.model_names
        flags_dict["msa_depth"] = args.msa_depth

    postprocess_flags = {
        "zip_pickles": args.compress_result_pickles,
        "remove_pickles": args.remove_result_pickles,
    }

    output_dir = args.output_directory
    if args.use_ap_style:
        output_dir = join(args.output_directory,object_to_model.description)
    makedirs(output_dir, exist_ok=True)
    return object_to_model, flags_dict, postprocess_flags,output_dir

def main():
    args = parse_args()

    data = create_custom_info(args.parsed_input)
    all_interactors = create_interactors(data, args.features_directory)
    objects_to_model = [] 
    for interactors in all_interactors:
        object_to_model, flags_dict, postprocess_flags, output_dir= pre_modelling_setup(interactors, args)
        objects_to_model.append({object_to_model : output_dir})

    predict_structure(
        multimeric_object=objects_to_model,
        model_flags=flags_dict,
        fold_backend=args.fold_backend,
        postprocess_flags=postprocess_flags
    )


if __name__ == "__main__":
    main()
