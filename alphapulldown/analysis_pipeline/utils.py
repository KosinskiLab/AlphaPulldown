from configparser import Interpolation
import IPython.display as display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import pandas as pd
from absl import logging
import subprocess
import sys
#from analysis_pipeline.af2hyde_mod import plot_predicted_alignment_error
from af2plots.plotter import plotter


def display_pae_plots(subdir,figsize=(50, 50)):
    """A function to display all the pae plots in the subdir"""
    pattern = r"ranked_(\d+)\.png"
    images = sorted([i for i in os.listdir(subdir) if ".png" in i],
                    key= lambda x: int(re.search(pattern,x).group(1)))
    if len(images) > 0:
        fig, axs = plt.subplots(1, len(images), figsize=figsize)
        for i in range(len(images)):
            img = plt.imread(os.path.join(subdir, images[i]))
            axs[i].imshow(img,interpolation="nearest")
            axs[i].axis("off")
        #plt.show()
    else:
        #plot_predicted_alignment_error(subdir)
        af2o = plotter()
        dd = af2o.parse_model_pickles(subdir)
        ff = af2o.plot_predicted_alignment_error(dd)

    plt.show()

def run_and_summarise_pi_score(work_dir, pdb_path, surface_thres: int = 2):
    """A function to calculate all predicted models' pi_scores and make a pandas df of the results"""

    subprocess.run(
    f"source activate pi_score && export PYTHONPATH=/software:$PYTHONPATH && python /software/pi_score/run_piscore_wc.py -p {pdb_path} -o {work_dir} -s {surface_thres} -ps 10", shell=True, executable='/bin/bash')
    
    output_df = pd.DataFrame()
    csv_files = [f for f in os.listdir(
        work_dir) if 'filter_intf_features' in f]
    pi_score_files = [f for f in os.listdir(work_dir) if 'pi_score_' in f]
    filtered_df = pd.read_csv(os.path.join(work_dir, csv_files[0]))

    if filtered_df.shape[0] == 0:
        for column in filtered_df.columns:
            filtered_df[column] = ["None"]
        filtered_df['jobs'] = str(job)
        filtered_df['pi_score'] = "No interface detected"
    else:
        with open(os.path.join(subdir, pi_score_files[0]), 'r') as f:
            lines = [l for l in f.readlines() if "#" not in l]
            if len(lines) > 0:
                pi_score = pd.read_csv(
                    os.path.join(subdir, pi_score_files[0]))
                pi_score['jobs'] = str(job)
            else:
                pi_score = pd.DataFrame.from_dict(
                    {"pi_score": ['SC:  mds: too many atoms']})
            f.close()
        filtered_df['jobs'] = str(job)
        pi_score['interface'] = pi_score['chains']
        filtered_df = pd.merge(filtered_df, pi_score, on=[
                                'jobs', 'interface'])
        try:
            filtered_df = filtered_df.drop(
                columns=["#PDB", "pdb", " pvalue", "chains", "predicted_class"])
        except:
            pass

    output_df = pd.concat([output_df, filtered_df])
    return output_df