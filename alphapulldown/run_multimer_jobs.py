#!/usr/bin/env python3

# Author: Dingquan Yu
# A script to create region information for create_multimer_features.py
# #

import itertools
from absl import app, logging
from alphapulldown.utils import (create_interactors, read_all_proteins, read_custom, make_dir_monomer_dictionary,
                                 load_monomer_objects, check_output_dir, create_model_runners_and_random_seed,
                                 create_and_save_pae_plots,post_prediction_process)
from itertools import combinations
from alphapulldown.objects import MultimericObject
import os
from pathlib import Path
from alphapulldown.predict_structure import predict, ModelsToRelax
from alphapulldown.utils import get_run_alphafold


run_af = get_run_alphafold()
flags = run_af.flags

flags.DEFINE_enum(
    "mode",
    "pulldown",
    ["pulldown", "all_vs_all", "homo-oligomer", "custom"],
    "choose the mode of running multimer jobs",
)
flags.DEFINE_string(
    "output_path", None, "output directory where the region data is going to be stored"
)
flags.DEFINE_string("oligomer_state_file", None, "path to oligomer state files")
flags.DEFINE_list(
    "monomer_objects_dir",
    None,
    "a list of directories where monomer objects are stored",
)
flags.DEFINE_list("protein_lists", None, "protein list files")

delattr(flags.FLAGS, "data_dir")
flags.DEFINE_string("data_dir", None, "Path to params directory")

flags.DEFINE_integer("num_cycle", 3, help="number of recycles")
flags.DEFINE_integer(
    "num_predictions_per_model", 1, "How many predictions per model. Default is 1"
)
flags.DEFINE_integer(
    "job_index", None, "index of sequence in the fasta file, starting from 1"
)
flags.DEFINE_boolean(
    "no_pair_msa", False, "do not pair the MSAs when constructing multimer objects"
)
flags.DEFINE_boolean(
    "multimeric_mode",
    False,
    "Run with multimeric template ",
)
flags.DEFINE_boolean(
    "gradient_msa_depth",
    False,
    "Run predictions for each model with logarithmically distributed MSA depth",
)
flags.DEFINE_string(
    "model_names", None, "Names of models to use, e.g. model_2_multimer_v3 (default: all models)"
)
flags.DEFINE_integer(
    "msa_depth", None, "Number of sequences to use from the MSA (by default is taken from AF model config)"
)
flags.DEFINE_boolean(
    "use_unifold",False,"Whether unifold models are going to be used. Default it False"
)

flags.DEFINE_boolean(
    "use_alphalink",False,"Whether alphalink models are going to be used. Default it False"
)
flags.DEFINE_string(
    "crosslinks",None,"Path to crosslink information pickle"
)
flags.DEFINE_string(
    "alphalink_weight",None,'Path to AlphaLink neural network weights'
)
flags.DEFINE_string(
    "unifold_param",None,'Path to UniFold neural network weights'
)
flags.DEFINE_boolean(
    "compress_result_pickles",False,"Whether the result pickles are going to be gzipped. Default False"
)
flags.DEFINE_boolean(
    "remove_result_pickles",False,"Whether the result pickles that do not belong to the best model are going to be removed. Default is False"
)
flags.DEFINE_enum("unifold_model_name","multimer_af2",
                  ["multimer_af2","multimer_ft","multimer","multimer_af2_v3","multimer_af2_model45_v3"],"choose unifold model structure")
flags.mark_flag_as_required("output_path")

delattr(flags.FLAGS, "models_to_relax")
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.NONE, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')

unused_flags = (
    'bfd_database_path',
    'db_preset',
    'fasta_paths',
    'hhblits_binary_path',
    'hhsearch_binary_path',
    'hmmbuild_binary_path',
    'hmmsearch_binary_path',
    'jackhmmer_binary_path',
    'kalign_binary_path',
    'max_template_date',
    'mgnify_database_path',
    'num_multimer_predictions_per_model',
    'obsolete_pdbs_path',
    'output_dir',
    'pdb70_database_path',
    'pdb_seqres_database_path',
    'small_bfd_database_path',
    'template_mmcif_dir',
    'uniprot_database_path',
    'uniref30_database_path',
    'uniref90_database_path',
)

for flag in unused_flags:
    delattr(flags.FLAGS, flag)

FLAGS = flags.FLAGS

def create_pulldown_info(
        bait_proteins: list, candidate_proteins: list, job_index=None
) -> dict:
    """
    A function to create apms info

    Args:
    all_proteins: list of all proteins in the fasta file parsed by read_all_proteins()
    bait_protein: name of the bait protein
    job_index: whether there is a job_index specified or not
    """
    all_protein_pairs = list(itertools.product(*[bait_proteins, *candidate_proteins]))
    num_cols = len(candidate_proteins) + 1
    data = dict()


    if job_index is None:
        for i in range(num_cols):
            curr_col = []
            for pair in all_protein_pairs:
                curr_col.append(pair[i])
            update_dict = {f"col_{i + 1}": curr_col}
            data.update(update_dict)


    elif isinstance(job_index, int):
        target_pair = all_protein_pairs[job_index - 1]
        for i in range(num_cols):
            update_dict = {f"col_{i + 1}": [target_pair[i]]}
            data.update(update_dict)
    return data




def create_all_vs_all_info(all_proteins: list, job_index=None):
    """A function to create all against all i.e. every possible pair of interaction"""
    all_possible_pairs = list(combinations(all_proteins, 2))
    if job_index is not None:
        job_index = job_index - 1
        combs = [all_possible_pairs[job_index-1]]
    else:
        combs = all_possible_pairs


    col1 = []
    col2 = []
    for comb in combs:
        col1.append(comb[0])
        col2.append(comb[1])


    data = {"col1": col1, "col2": col2}
    return data




def create_custom_info(all_proteins):
    """
    A function to create 'data' for custom input file
    """
    num_cols = len(all_proteins)
    data = dict()
    for i in range(num_cols):
        data[f"col_{i+1}"] = [all_proteins[i]]
    return data

def create_multimer_objects(data, monomer_objects_dir, pair_msa=True):
    """
    A function to create multimer objects

    Arg
    data: a dictionary created by create_all_vs_all_info() or create_apms_info()
    monomer_objects_dir: a directory where pre-computed monomer objects are stored
    """
    multimers = []
    num_jobs = len(data[list(data.keys())[0]])
    job_idxes = list(range(num_jobs))
    import glob
    path = os.path.join(monomer_objects_dir[0],'*.pkl')
    pickles = set([os.path.basename(fl) for fl in glob.glob(path)])
    required_pickles = set(key+".pkl" for value_list in data.values()
                    for value_dict in value_list
                    for key in value_dict.keys())
    missing_pickles = required_pickles.difference(pickles)
    if len(missing_pickles) > 0:
        raise Exception(f"{missing_pickles} not found in {monomer_objects_dir}")
    else:
        logging.info("All pickle files have been found")


    for job_idx in job_idxes:
        interactors = create_interactors(data, monomer_objects_dir, job_idx)
        if len(interactors) > 1:
            multimer = MultimericObject(interactors=interactors,pair_msa=pair_msa, multimeric_mode = FLAGS.multimeric_mode)
            logging.info(f"done creating multimer {multimer.description}")
            multimers.append(multimer)
        else:
            logging.info(f"done loading monomer {interactors[0].description}")
            multimers.append(interactors[0])
    return multimers




def create_homooligomers(oligomer_state_file, monomer_objects_dir, job_index=None, pair_msa = False):
    """a function to read homooligomer state"""
    multimers = []
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    with open(oligomer_state_file) as f:
        lines = list(f.readlines())
        if job_index is not None:
            job_idxes = [job_index - 1]
        else:
            job_idxes = list(range(len(lines)))


        for job_idx in job_idxes:
            l = lines[job_idx]
            if len(l.strip()) > 0:
                if len(l.rstrip().split(",")) > 1:
                    protein_name = l.rstrip().split(",")[0]
                    num_units = int(l.rstrip().split(",")[1])
                else:
                    protein_name = l.rstrip().split(",")[0]
                    num_units = 1


                if num_units > 1:
                    monomer = load_monomer_objects(monomer_dir_dict, protein_name)
                    interactors = [monomer] * num_units
                    homooligomer = MultimericObject(interactors,pair_msa=pair_msa)
                    homooligomer.description = f"{protein_name}_homo_{num_units}er"
                    multimers.append(homooligomer)
                    logging.info(
                        f"finished creating homooligomer {homooligomer.description}"
                    )
                elif num_units == 1:
                    monomer = load_monomer_objects(monomer_dir_dict, protein_name)
                    multimers.append(monomer)
                    logging.info(f"finished loading monomer: {protein_name}")
        f.close()
    return multimers




def create_custom_jobs(custom_input_file, monomer_objects_dir, job_index=None, pair_msa=True):
    """
    A function to create multimers under custom mode

    Args
    custom_input_file: A list of input_files from FLAGS.protein_lists

    """
    lines = []
    for file in custom_input_file:
        with open(file) as f:
            lines = lines + list(f.readlines())
            f.close()
    if job_index is not None:
        logging.info("Running in parallel mode")
        job_idxes = [job_index - 1]
    else:
        logging.info("Running in serial mode")
        job_idxes = list(range(len(lines)))
    multimers = []
    for job_idx in job_idxes:
        l = lines[job_idx]
        if len(l.strip()) > 0:
            all_proteins = read_custom(l)
            data = create_custom_info(all_proteins)
            multimer = create_multimer_objects(data, monomer_objects_dir, pair_msa=pair_msa)
            multimers += multimer
    return multimers




def predict_individual_jobs(multimer_object, output_path, model_runners, random_seed):
    output_path = os.path.join(output_path, multimer_object.description)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"now running prediction on {multimer_object.description}")
    logging.info(f"output_path is {output_path}")
    if not isinstance(multimer_object, MultimericObject):
        multimer_object.input_seqs = [multimer_object.sequence]

    if FLAGS.use_unifold:
        from unifold.inference import config_args,unifold_config_model,unifold_predict
        from unifold.dataset import process_ap
        from unifold.config import model_config
        configs = model_config(FLAGS.unifold_model_name)
        general_args = config_args(FLAGS.unifold_param,
                                   target_name=multimer_object.description,
                                   output_dir=output_path)
        model_runner = unifold_config_model(general_args)
        # First need to add num_recycling_iters to the feature dictionary
        # multimer_object.feature_dict.update({"num_recycling_iters":general_args.max_recycling_iters})
        processed_features,_ = process_ap(config=configs.data,
                                          features=multimer_object.feature_dict,
                                          mode="predict",labels=None,
                                          seed=42,batch_idx=None,
                                          data_idx=None,is_distillation=False
                                          )
        logging.info(f"finished configuring the Unifold AlphlaFcd old model and process numpy features")
        unifold_predict(model_runner,general_args,processed_features)

    elif FLAGS.use_alphalink:
        assert FLAGS.alphalink_weight is not None
        from unifold.alphalink_inference import alphalink_prediction
        from unifold.config import model_config
        logging.info(f"Start using AlphaLink weights and cross-link information")  
        MODEL_NAME = 'model_5_ptm_af2'
        configs = model_config(MODEL_NAME)
        alphalink_prediction(multimer_object.feature_dict,
                             os.path.join(FLAGS.output_path,multimer_object.description),
                             input_seqs = multimer_object.input_seqs,
                             param_path = FLAGS.alphalink_weight,
                             configs = configs,crosslinks=FLAGS.crosslinks,
                             chain_id_map=multimer_object.chain_id_map)
    else:
        predict(
            model_runners,
            output_path,
            multimer_object.feature_dict,
            random_seed,
            FLAGS.benchmark,
            fasta_name=multimer_object.description,
            models_to_relax=FLAGS.models_to_relax,
            seqs=multimer_object.input_seqs,
        )
        create_and_save_pae_plots(multimer_object, output_path)
        post_prediction_process(output_path,
                           zip_pickles = FLAGS.compress_result_pickles,
                           remove_pickles = FLAGS.remove_result_pickles
                           )

def predict_multimers(multimers):
    """
    Final function to predict multimers

    Args
    multimers: A list of multimer objects created by create_multimer_objects()
    or create_custom_jobs() or create_homooligomers()
    """
    for object in multimers:
        logging.info('object: '+object.description)
        path_to_models = os.path.join(FLAGS.output_path, object.description)
        logging.info(f"Modeling new interaction for {path_to_models}")
        if isinstance(object, MultimericObject):
            model_runners, random_seed = create_model_runners_and_random_seed(
                "multimer",
                FLAGS.num_cycle,
                FLAGS.random_seed,
                FLAGS.data_dir,
                FLAGS.num_predictions_per_model,
                FLAGS.gradient_msa_depth,
                FLAGS.model_names,
                FLAGS.msa_depth,
            )
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )
        else:
            model_runners, random_seed = create_model_runners_and_random_seed(
                "monomer_ptm",
                FLAGS.num_cycle,
                FLAGS.random_seed,
                FLAGS.data_dir,
                FLAGS.num_predictions_per_model,
            )
            logging.info("will run in monomer mode")
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )

def main(argv):
    check_output_dir(FLAGS.output_path)
    if FLAGS.mode == "pulldown":
        bait_proteins = read_all_proteins(FLAGS.protein_lists[0])
        candidate_proteins = []
        for file in FLAGS.protein_lists[1:]:
            candidate_proteins.append(read_all_proteins(file))
        data = create_pulldown_info(
            bait_proteins, candidate_proteins, job_index=FLAGS.job_index
        )
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir, not FLAGS.no_pair_msa)


    elif FLAGS.mode == "all_vs_all":
        all_proteins = read_all_proteins(FLAGS.protein_lists[0])
        data = create_all_vs_all_info(all_proteins, job_index=FLAGS.job_index)
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir, not FLAGS.no_pair_msa)


    elif FLAGS.mode == "homo-oligomer":
        multimers = create_homooligomers(
            FLAGS.oligomer_state_file,
            FLAGS.monomer_objects_dir,
            job_index=FLAGS.job_index,
            pair_msa=not FLAGS.no_pair_msa
        )


    elif FLAGS.mode == "custom":
        multimers = create_custom_jobs(
            FLAGS.protein_lists, FLAGS.monomer_objects_dir, job_index=FLAGS.job_index, pair_msa=not FLAGS.no_pair_msa
        )

    predict_multimers(multimers)




if __name__ == "__main__":
    app.run(main)
