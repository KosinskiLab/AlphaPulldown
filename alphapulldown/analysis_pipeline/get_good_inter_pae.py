#!/usr/bin/env python3

from calculate_mpdockq import get_best_plddt, read_pdb, read_plddt, score_complex, calculate_mpDockQ, read_pdb_pdockq, calc_pdockq
from pdb_analyser import PDBAnalyser
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import os
import pickle
from absl import flags, app, logging
import json
import numpy as np
import pandas as pd
import gzip
from typing import Tuple

flags.DEFINE_string('output_dir', None,
                    'directory where predicted models are stored')
flags.DEFINE_float(
    'cutoff', 5.0, 'cutoff value of PAE. i.e. only pae<cutoff is counted good')
flags.DEFINE_integer('surface_thres', 2, 'surface threshold. must be integer')
FLAGS = flags.FLAGS


def examine_inter_pae(pae_mtx, seq_lengths, cutoff):
    """A function that checks inter-pae values in multimer prediction jobs"""
    old_length = 0
    mtx = pae_mtx.copy()
    for length in seq_lengths:
        new_length = old_length + length
        mtx[old_length:new_length, old_length:new_length] = 50
        old_length = new_length
    check = np.where(mtx < cutoff)[0].size != 0

    return check


def obtain_mpdockq(work_dir):
    """Returns mpDockQ if more than two chains otherwise return pDockQ"""
    pdb_path = os.path.join(work_dir, 'ranked_0.pdb')
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_path)
    best_plddt = get_best_plddt(work_dir)
    plddt_per_chain = read_plddt(best_plddt, chain_CA_inds)
    complex_score, num_chains = score_complex(
        chain_coords, chain_CB_inds, plddt_per_chain)
    if complex_score is not None and num_chains > 2:
        mpDockq_or_pdockq = calculate_mpDockQ(complex_score)
    elif complex_score is not None and num_chains == 2:
        chain_coords, plddt_per_chain = read_pdb_pdockq(pdb_path)
        mpDockq_or_pdockq = calc_pdockq(chain_coords, plddt_per_chain, t=8)
    else:
        mpDockq_or_pdockq = "None"
    return mpDockq_or_pdockq, plddt_per_chain

def obtain_pae_and_iptm(result_subdir: str, best_model: str) -> Tuple[np.array, float]:
    """A function that obtains PAE matrix and iptm score from either results pickle, zipped pickle, or pae_json"""
    try:
        ranking_results = json.load(
            open(os.path.join(result_subdir, "ranking_debug.json")))
    except FileNotFoundError as e:
        logging.warning(f"ranking_debug.json is not found at {result_subdir}")
    iptm_score = "None"
    pae_mtx = None
    if "iptm" in ranking_results:
        iptm_score = ranking_results['iptm'].get(best_model, None)

    if os.path.exists(f"{result_subdir}/pae_{best_model}.json"):
        pae_results = json.load(
            open(f"{result_subdir}/pae_{best_model}.json"))[0]['predicted_aligned_error']
        pae_mtx = np.array(pae_results)

    if iptm_score == "None":
        try:
            check_dict = pickle.load(
                open(os.path.join(result_subdir, f"result_{best_model}.pkl"), 'rb'))
            iptm_score = check_dict['iptm']
            pae_mtx = check_dict['predicted_aligned_error']
        except FileNotFoundError:
            logging.error(os.path.join(
                result_subdir, f"result_{best_model}.pkl")+" does not exist. Will search for pkl.gz")
            try:
                check_dict = pickle.load(gzip.open(os.path.join(
                result_subdir, f"result_{best_model}.pkl.gz"), 'rb'))
                iptm_score = check_dict['iptm']
                pae_mtx = check_dict['predicted_aligned_error']
            except FileNotFoundError:
                logging.error(
                    os.path.join(
                result_subdir, f"result_{best_model}.pkl.gz")+" does not exist. Failed to extract iptm score."
                )
    return pae_mtx, iptm_score


def obtain_seq_lengths(result_subdir: str) -> list:
    """A function that obtain length of each chain in ranked_0.pdb"""
    best_result_pdb = os.path.join(result_subdir, "ranked_0.pdb")
    seq_length = []
    pdb_parser = PDBParser()
    sequence_builder = PPBuilder()
    if not os.path.exists(best_result_pdb):
        raise FileNotFoundError(
            f"ranked_0.pdb is not found at {result_subdir}")
    else:
        structure = pdb_parser.get_structure("ranked_0", best_result_pdb)
        seqs = sequence_builder.build_peptides(structure)
        for seq in seqs:
            seq_length.append(len(seq.get_sequence()))
    return seq_length


def main(argv):
    jobs = os.listdir(FLAGS.output_dir)
    count = 0
    good_jobs = []
    output_df = pd.DataFrame()
    pi_score_output_path = os.path.join(FLAGS.output_dir, "pi_score_outputs")
    os.makedirs(pi_score_output_path,exist_ok=True)
    for job in jobs:
        try:
            count += 1
            logging.info(f"now processing {job}")
            ranking_debug_path = os.path.join(FLAGS.output_dir, job, 'ranking_debug.json')
            
            if os.path.isfile(ranking_debug_path):
                try:
                    pdb_analyser = PDBAnalyser(os.path.join(FLAGS.output_dir, job, "ranked_0.pdb"))
                    result_subdir = os.path.join(FLAGS.output_dir, job)
                    
                    with open(ranking_debug_path, 'r') as f:
                        ranking_data = json.load(f)
                    try:
                        best_model = ranking_data['order'][0]
                        data = ranking_data

                        if "iptm+ptm" in data:
                            iptm_ptm_score = data['iptm+ptm'][best_model]
                            
                            try:
                                pae_mtx, iptm_score = obtain_pae_and_iptm(result_subdir, best_model)
                                seq_lengths = obtain_seq_lengths(result_subdir)
                                check = examine_inter_pae(pae_mtx, seq_lengths, cutoff=FLAGS.cutoff)
                                mpDockq_score, plddt_per_chain = obtain_mpdockq(os.path.join(FLAGS.output_dir, job))
                                
                                if check:
                                    good_jobs.append(job)
                                    score_df = pdb_analyser(os.path.join(pi_score_output_path, job), pae_mtx, plddt_per_chain)
                                    score_df['jobs'] = job
                                    score_df['iptm_ptm'] = iptm_ptm_score
                                    score_df['iptm'] = iptm_score
                                    score_df['pDockQ/mpDockQ'] = mpDockq_score
                                    
                                    for i in ['pDockQ/mpDockQ', 'iptm', 'iptm_ptm', 'jobs']:
                                        score_df.insert(0, i, score_df.pop(i))
                                        
                                    output_df = pd.concat([score_df, output_df])
                            except Exception as e:
                                logging.error(f"Error processing PAE and iPTM for job {job}: {e}")
                        else:
                            logging.warning(f"{job} does not seem to be a multimeric model. iptm+ptm scores are not in the ranking_debug.json Skipped")
                    except Exception as e:
                        logging.error(f"Error getting the best model name from ranking_debug.json for job :{job} : {e}")
                except Exception as e:
                    logging.error(f"Error processing ranking_debug.json for job {job}: {e}")
            else:
                logging.warning(f"{job} does not have ranking_debug.json. Skipped.")
        except Exception as e:
            logging.error(f"Error processing job {job}: {e}")
        finally:
            logging.info(f"done for {job} {count} out of {len(jobs)} finished.")

    if len(good_jobs) == 0:
        logging.info(
            f"Unfortunately, none of your protein models had at least one PAE on the interface below your cutoff value : {FLAGS.cutoff}.\n Please consider using a larger cutoff.")
    else:
        unwanted_columns = ['pdb',' pvalue', 'pvalue']
        for c in unwanted_columns:
            if c in output_df:
                output_df = output_df.drop(columns=c)
        output_df = output_df.sort_values(by='iptm', ascending= False)
        if "Hydrophobhic" in output_df.columns:
            output_df = output_df.rename(columns={"Hydrophobhic" : "Hydrophobic"})
        output_df.to_csv(os.path.join(FLAGS.output_dir,"predictions_with_good_interpae.csv"),index=False)

if __name__ == '__main__':
    app.run(main)
