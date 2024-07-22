#!/usr/bin/env python3

# This script is to obtain all predicted multimer jobs
# with good
#  #

from math import pi
from operator import index
import os
import pickle
from absl import flags, app, logging
import json
import numpy as np
import pandas as pd
import subprocess
import gzip
from analysis_pipeline.utils import obtain_seq_lengths,obtain_pae_and_iptm

flags.DEFINE_string("output_dir", '.', "directory where predicted models are stored")
flags.DEFINE_float(
    "cutoff", 5.0, "cutoff value of PAE. i.e. only pae<cutoff is counted good"
)
flags.DEFINE_boolean("create_notebook", True, "Whether creating a notebook")
flags.DEFINE_integer("surface_thres", 2, "surface threshold. must be integer")
flags.DEFINE_integer("pae_figsize",50,"figsize of pae_plot, default is 50")
FLAGS = flags.FLAGS


def examine_inter_pae(pae_mtx, seq_lengths, cutoff):
    """A function that checks inter-pae values in multimer prediction jobs"""
    old_lenth = 0
    for length in seq_lengths:
        new_length = old_lenth + length
        pae_mtx[old_lenth:new_length, old_lenth:new_length] = 50
        old_lenth = new_length
    check = np.where(pae_mtx < cutoff)[0].size != 0

    return check


def create_notebook(combo, output_dir,figsize):
    from nbformat import current as nbf

    nb = nbf.new_notebook()
    output_cells = []
    md_cell = nbf.new_text_cell(
        "markdown",
        "# A notebook to display all the predictions with good inter-pae scores",
    )
    import_cell = nbf.new_code_cell(
        "from analysis_pipeline.af2_3dmol import parse_results,parse_results_colour_chains"
    )
    disable_autosave_cell = nbf.new_code_cell(f"%autosave 0")
    output_cells.append(md_cell)
    output_cells.append(disable_autosave_cell)
    output_cells.append(import_cell)

    import_cell = nbf.new_code_cell(
        "from analysis_pipeline.utils import display_pae_plots"
    )
    output_cells.append(import_cell)
    base_dir = output_dir
    for i in range(combo.shape[0]):
        job = combo.iloc[i, 0]
        iptm_score = combo.iloc[i, -1]
        title_cell = nbf.new_text_cell("markdown", f"## {job} with iptm: {iptm_score}")
        output_cells.append(title_cell)
        subdir = os.path.join(base_dir, f"{job}")
        subtitile1 = nbf.new_text_cell("markdown", f"### {job} PAE plots")
        output_cells.append(subtitile1)
        code_cell_1 = nbf.new_code_cell(f"display_pae_plots('{subdir}',figsize=({figsize,figsize}))")
        output_cells.append(code_cell_1)
        subtitle2 = nbf.new_text_cell("markdown", f"### {job} coloured by plddt")
        output_cells.append(subtitle2)

        code_cell_2 = nbf.new_code_cell(f"parse_results('{subdir}')")
        output_cells.append(code_cell_2)
        subtitile3 = nbf.new_text_cell("markdown", f"### {job} coloured by chains")
        output_cells.append(subtitile3)
        code_cell_3 = nbf.new_code_cell(f"parse_results_colour_chains('{subdir}')")
        output_cells.append(code_cell_3)
    nb["worksheets"].append(nbf.new_worksheet(cells=output_cells))
    with open(os.path.join(output_dir, "output.ipynb"), "w") as f:
        nbf.write(nb, f, "ipynb", version=4)
    logging.info("A notebook has been successfully created.")
    # logging.info("A notebook has been successfully created. Now will execute the notebook")
    # subprocess.run(f"source  activate programme_notebook && jupyter nbconvert --to notebook --inplace --execute {output_dir}/output.ipynb",shell=True,executable='/bin/bash')


def main(argv):
    jobs = os.listdir(FLAGS.output_dir)
    good_jobs = []
    iptm_ptm = []
    iptm = []
    count = 0
    for job in jobs:
        logging.info(f"now processing {job}")
        count = count + 1
        if os.path.isfile(os.path.join(FLAGS.output_dir, job, "ranking_debug.json")):
            result_subdir = os.path.join(FLAGS.output_dir, job)
            best_result = json.load(open(os.path.join(FLAGS.output_dir, job, "ranking_debug.json")))['order'][0]
            best_result = "result_" + best_result
            seq_lengths = obtain_seq_lengths(os.path.join(FLAGS.output_dir, job))    
            
            best_model = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), "rb")
            )["order"][0]

            # below first check whether it is a job in monomer mode or multimer
            data = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), "rb")
            )

            if "iptm+ptm" in data.keys():
                iptm_ptm_score = data["iptm+ptm"][best_model]   
                pae_mtx,iptm_score = obtain_pae_and_iptm(result_subdir=os.path.join(FLAGS.output_dir, job),
                                              best_model=best_model)
                check = examine_inter_pae(pae_mtx, seq_lengths, cutoff=FLAGS.cutoff)
                if check:
                    good_jobs.append(str(job))
                    iptm_ptm.append(iptm_ptm_score)
                    iptm.append(iptm_score)
        logging.info(f"done for {job} {count} out of {len(jobs)} finished.")

    pi_score_df = pd.DataFrame()
    pi_score_df["jobs"] = good_jobs
    pi_score_df["iptm+ptm"] = iptm_ptm
    pi_score_df["iptm"] = iptm

    pi_score_df = pi_score_df.sort_values(by="iptm", ascending=False)

    if FLAGS.create_notebook:
        create_notebook(pi_score_df, FLAGS.output_dir,FLAGS.pae_figsize)


if __name__ == "__main__":
    app.run(main)
