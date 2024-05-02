#!/usr/bin/env python3

from calculate_mpdockq import *
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
    old_lenth = 0
    mtx = pae_mtx.copy()
    for length in seq_lengths:
        new_length = old_lenth + length
        mtx[old_lenth:new_length, old_lenth:new_length] = 50
        old_lenth = new_length
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
        except FileNotFoundError:
            logging.info(os.path.join(
                result_subdir, f"result_{best_model}.pkl")+" does not exist. Will search for pkl.gz")
            check_dict = pickle.load(gzip.open(os.path.join(
                result_subdir, f"result_{best_model}.pkl.gz"), 'rb'))
        finally:
            logging.info(f"finished reading results for the best model.")
            pae_mtx = check_dict['predicted_aligned_error']
            iptm_score = check_dict['iptm']
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
    for job in jobs:
        count = count + 1
        logging.info(f"now processing {job}")
        if os.path.isfile(os.path.join(FLAGS.output_dir, job, 'ranking_debug.json')):
            pdb_analyser = PDBAnalyser(os.path.join(
                FLAGS.output_dir, job, "ranked_0.pdb"))         
            result_subdir = os.path.join(FLAGS.output_dir, job)
            best_model = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), 'r'))['order'][0]
            data = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), 'r'))
            if "iptm" in data.keys() and "iptm+ptm" in data.keys():
                iptm_ptm_score = data['iptm+ptm'][best_model]
                pae_mtx, iptm_score = obtain_pae_and_iptm(
                    result_subdir, best_model)
                seq_lengths = obtain_seq_lengths(result_subdir)
                check = examine_inter_pae(
                    pae_mtx, seq_lengths, cutoff=FLAGS.cutoff)
                mpDockq_score, plddt_per_chain = obtain_mpdockq(
                    os.path.join(FLAGS.output_dir, job))
                if check:
                    good_jobs.append(job)
                    score_df = pdb_analyser(
                        pae_mtx, plddt_per_chain)
                    score_df['jobs']=job
                    score_df['iptm_ptm'] = iptm_ptm_score
                    score_df['iptm'] = iptm_score
                    score_df['pDockQ/mpDockQ'] = mpDockq_score
                    for i in ['pDockQ/mpDockQ', 'iptm', 'iptm_ptm','jobs']:
                        score_df.insert(0, i, score_df.pop(i))
                    output_df = pd.concat([score_df,output_df])
        else:
            logging.warning(f"{job} does not have ranking_debug.json. Skipped.")
        logging.info(
            f"done for {job} {count} out of {len(jobs)} finished.")
    if len(good_jobs) == 0:
        logging.info(
            f"Unfortunately, none of your protein models had at least one PAE on the interface below your cutoff value : {FLAGS.cutoff}.\n Please consider using a larger cutoff.")
    else:
        output_df.to_csv(os.path.join(FLAGS.output_dir,"predictions_with_good_interpae.csv"),index=False)

if __name__ == '__main__':
    app.run(main)
