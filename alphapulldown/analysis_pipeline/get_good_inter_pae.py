#!/usr/bin/env python3

from math import pi
from operator import index
import os 
import pickle
from absl import flags,app,logging
import json
import numpy as np
import pandas as pd
import subprocess
from calculate_mpdockq import *
import sys 
import gzip

flags.DEFINE_string('output_dir',None,'directory where predicted models are stored')
flags.DEFINE_float('cutoff',5.0,'cutoff value of PAE. i.e. only pae<cutoff is counted good')
flags.DEFINE_integer('surface_thres',2,'surface threshold. must be integer')
FLAGS=flags.FLAGS

def examine_inter_pae(pae_mtx,seqs,cutoff):
    """A function that checks inter-pae values in multimer prediction jobs"""
    lens = [len(seq) for seq in seqs]
    old_lenth=0
    for length in lens:
        new_length = old_lenth + length
        pae_mtx[old_lenth:new_length,old_lenth:new_length] = 50
        old_lenth = new_length
    check = np.where(pae_mtx<cutoff)[0].size !=0

    return check


def obtain_mpdockq(work_dir):
    """Returns mpDockQ if more than two chains otherwise return pDockQ"""
    pdb_path = os.path.join(work_dir,'ranked_0.pdb')
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_path)
    best_plddt = get_best_plddt(work_dir)
    plddt_per_chain = read_plddt(best_plddt,chain_CA_inds)
    complex_score,num_chains = score_complex(chain_coords,chain_CB_inds,plddt_per_chain)
    if complex_score is not None and num_chains>2:
        mpDockq_or_pdockq = calculate_mpDockQ(complex_score)
    elif complex_score is not None and num_chains==2:
        chain_coords,plddt_per_chain = read_pdb_pdockq(pdb_path)
        mpDockq_or_pdockq = calc_pdockq(chain_coords,plddt_per_chain,t=8)
    else:
        mpDockq_or_pdockq = "None"
    return mpDockq_or_pdockq

def run_and_summarise_pi_score(workd_dir,jobs,surface_thres):

    """A function to calculate all predicted models' pi_scores and make a pandas df of the results"""
    try:
        os.remove(f"mkdir {workd_dir}/pi_score_outputs")
    except:
        pass
    subprocess.run(f"mkdir {workd_dir}/pi_score_outputs",shell=True,executable='/bin/bash')
    pi_score_outputs = os.path.join(workd_dir,"pi_score_outputs")
    for job in jobs:
        subdir = os.path.join(workd_dir,job)
        if not os.path.isfile(os.path.join(subdir,"ranked_0.pdb")):
            print(f"{job} failed. Cannot find ranked_0.pdb in {subdir}")
            sys.exit()
        else:
            pdb_path = os.path.join(subdir,"ranked_0.pdb")
            output_dir = os.path.join(pi_score_outputs,f"{job}")
            logging.info(f"pi_score output for {job} will be stored at {output_dir}")
            subprocess.run(f"source activate pi_score && export PYTHONPATH=/software:$PYTHONPATH && python /software/pi_score/run_piscore_wc.py -p {pdb_path} -o {output_dir} -s {surface_thres} -ps 10",shell=True,executable='/bin/bash')
            

    output_df = pd.DataFrame()
    for job in jobs:
        subdir = os.path.join(pi_score_outputs,job)
        csv_files = [f for f in os.listdir(subdir) if 'filter_intf_features' in f]
        pi_score_files = [f for f in os.listdir(subdir) if 'pi_score_' in f]
        filtered_df = pd.read_csv(os.path.join(subdir,csv_files[0]))
    
        if filtered_df.shape[0]==0:
            for column in filtered_df.columns:
                filtered_df[column] = ["None"]
            filtered_df['jobs'] = str(job)
            filtered_df['pi_score'] = "No interface detected"
        else:
            with open(os.path.join(subdir,pi_score_files[0]),'r') as f:
                lines = [l for l in f.readlines() if "#" not in l]
                if len(lines)>0:
                    pi_score = pd.read_csv(os.path.join(subdir,pi_score_files[0]))
                    pi_score['jobs']=str(job)
                else:
                    pi_score = pd.DataFrame.from_dict({"pi_score":['SC:  mds: too many atoms']})
                f.close()
            filtered_df['jobs'] = str(job)
            pi_score['interface'] = pi_score['chains']
            filtered_df=pd.merge(filtered_df,pi_score,on=['jobs','interface'])
            try:
                filtered_df=filtered_df.drop(columns=["#PDB","pdb"," pvalue","chains","predicted_class"])
            except:
                pass
        
        output_df = pd.concat([output_df,filtered_df])
    return output_df
    
    

def main(argv):
    jobs = os.listdir(FLAGS.output_dir)
    good_jobs = []
    iptm_ptm = list()
    iptm = list()
    mpDockq_scores = list()
    count = 0
    for job in jobs:
        logging.info(f"now processing {job}")
        if os.path.isfile(os.path.join(FLAGS.output_dir,job,'ranking_debug.json')):
            count=count +1
            result_subdir = os.path.join(FLAGS.output_dir,job)
            best_model = json.load(open(os.path.join(result_subdir,"ranking_debug.json"),'rb'))['order'][0]
            data = json.load(open(os.path.join(result_subdir,"ranking_debug.json"),'rb'))
            if "iptm" in data.keys() or "iptm+ptm" in data.keys():
                iptm_ptm_score = data['iptm+ptm'][best_model]
                try:
                    check_dict = pickle.load(open(os.path.join(result_subdir,f"result_{best_model}.pkl"),'rb'))
                except FileNotFoundError:
                    print(os.path.join(result_subdir,f"result_{best_model}.pkl")+" does not exist. Will search for pkl.gz")
                    check_dict = pickle.load(gzip.open(os.path.join(result_subdir,f"result_{best_model}.pkl.gz"),'rb'))
                finally:
                    print(f"finished reading result pickle for the best model.")
                seqs = check_dict['seqs']
                iptm_score = check_dict['iptm']
                pae_mtx = check_dict['predicted_aligned_error']
                check = examine_inter_pae(pae_mtx,seqs,cutoff=FLAGS.cutoff)
                mpDockq_score = obtain_mpdockq(os.path.join(FLAGS.output_dir,job))
                if check:
                    good_jobs.append(str(job))
                    iptm_ptm.append(iptm_ptm_score)
                    iptm.append(iptm_score)
                    mpDockq_scores.append(mpDockq_score)
            logging.info(f"done for {job} {count} out of {len(jobs)} finished.")
    other_measurements_df=pd.DataFrame.from_dict({
        "jobs":good_jobs,
        "iptm_ptm":iptm_ptm,
        "iptm":iptm,
        "mpDockQ/pDockQ":mpDockq_scores
    })
    pi_score_df = run_and_summarise_pi_score(FLAGS.output_dir,good_jobs,FLAGS.surface_thres)
    pi_score_df=pd.merge(pi_score_df,other_measurements_df,on="jobs")
    columns = list(pi_score_df.columns.values)
    columns.pop(columns.index('jobs'))
    pi_score_df = pi_score_df[['jobs'] + columns]
    pi_score_df = pi_score_df.sort_values(by='iptm',ascending=False)
    
    pi_score_df.to_csv(os.path.join(FLAGS.output_dir,"predictions_with_good_interpae.csv"),index=False)
    

if __name__ =='__main__':
    app.run(main)