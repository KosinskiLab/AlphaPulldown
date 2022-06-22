#
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
from calculate_mpdockq import *

flags.DEFINE_string("output_dir", None, "directory where predicted models are stored")
flags.DEFINE_float(
    "cutoff", 5.0, "cutoff value of PAE. i.e. only pae<cutoff is counted good"
)
flags.DEFINE_boolean("create_notebook", False, "Whether creating a notebook")
flags.DEFINE_integer("surface_thres", 2, "surface threshold. must be integer")
FLAGS = flags.FLAGS


def examine_inter_pae(pae_mtx, seqs, cutoff):
    """A function that checks inter-pae values in multimer prediction jobs"""
    lens = [len(seq) for seq in seqs]
    old_lenth = 0
    for length in lens:
        new_length = old_lenth + length
        pae_mtx[old_lenth:new_length, old_lenth:new_length] = 50
        old_lenth = new_length
    check = np.where(pae_mtx < cutoff)[0].size != 0

    return check


def create_notebook(combo, output_dir):
    from nbformat import current as nbf

    nb = nbf.new_notebook()
    output_cells = []
    md_cell = nbf.new_text_cell(
        "markdown",
        "# A notebook to display all the predictions with good inter-pae scores",
    )
    import_cell = nbf.new_code_cell(
        "from programme_notebook.af2hyde_mod import parse_results,parse_results_colour_chains"
    )
    output_cells.append(md_cell)
    output_cells.append(import_cell)
    import_cell = nbf.new_code_cell(
        "from programme_notebook.utils import display_pae_plots"
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
        code_cell_1 = nbf.new_code_cell(f"display_pae_plots('{subdir}')")
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
    logging.info(
        "A notebook has been successfully created. Now will execute the notebook"
    )
    subprocess.run(
        f"source  activate programme_notebook && jupyter nbconvert --to notebook --inplace --execute {output_dir}/output.ipynb",
        shell=True,
        executable="/bin/bash",
    )


def obtain_mpdockq(work_dir):
    pdb_path = os.path.join(work_dir, "ranked_0.pdb")
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_path)
    best_plddt = get_best_plddt(work_dir)
    plddt_per_chain = read_plddt(best_plddt, chain_CA_inds)
    complex_score = score_complex(chain_coords, chain_CB_inds, plddt_per_chain)
    if complex_score is not None:
        mpDockq = calculate_mpDockQ(complex_score)
    else:
        mpDockq = "None"
    return mpDockq


def run_and_summarise_pi_score(workd_dir, jobs, surface_thres):
    """A function to calculate all predicted models' pi_scores and make a pandas df of the results"""
    subprocess.run(
        f"mkdir {workd_dir}/pi_score_outputs", shell=True, executable="/bin/bash"
    )
    pi_score_outputs = os.path.join(workd_dir, "pi_score_outputs")
    for job in jobs:
        subdir = os.path.join(workd_dir, job)
        try:
            os.path.isfile(os.path.join(subdir, "ranked_0.pdb"))
            pdb_path = os.path.join(subdir, "ranked_0.pdb")
            output_dir = os.path.join(pi_score_outputs, f"{job}")
            subprocess.run(
                f"source activate pi_score && export PYTHONPATH=/software:$PYTHONPATH && python /software/pi_score/run_piscore_wc.py -p {pdb_path} -o {output_dir} -s {surface_thres} -ps 1",
                shell=True,
                executable="/bin/bash",
            )
        except FileNotFoundError:
            print(f"{job} failed. Cannot find ranked_0.pdb in {subdir}")

    output_df = pd.DataFrame()
    for job in jobs:
        subdir = os.path.join(pi_score_outputs, job)
        csv_files = [f for f in os.listdir(subdir) if "filter_intf_features" in f]
        pi_score_files = [f for f in os.listdir(subdir) if "pi_score_" in f]
        filtered_df = pd.read_csv(os.path.join(subdir, csv_files[0]))
        if filtered_df.shape[0] == 0:
            for column in filtered_df.columns:
                filtered_df[column] = ["None"]
            filtered_df["jobs"] = str(job)
            filtered_df["pi_score"] = "No interface detected"
        else:
            with open(os.path.join(subdir, pi_score_files[0]), "r") as f:
                lines = [l for l in f.readlines() if "#" not in l]
                if len(lines) > 0:
                    pi_score = float(lines[0].split(",")[-1])
                else:
                    pi_score = "SC:  mds: too many atoms"
                f.close()
            filtered_df["jobs"] = str(job)
            filtered_df["pi_score"] = pi_score

        output_df = pd.concat([output_df, filtered_df])
    output_df = output_df.drop(columns=["pdb"])
    return output_df


def main(argv):
    jobs = os.listdir(FLAGS.output_dir)
    good_jobs = []
    iptm_ptm = []
    iptm = []
    mpDockq_scores = []
    count = 0
    for job in jobs:
        logging.info(f"now processing {job}")
        if os.path.isfile(os.path.join(FLAGS.output_dir, job, "ranking_debug.json")):
            count = count + 1
            result_subdir = os.path.join(FLAGS.output_dir, job)
            best_result_pkl = sorted(
                [i for i in os.listdir(result_subdir) if "result_model_" in i]
            )[0]
            result_path = os.path.join(result_subdir, best_result_pkl)
            seqs = pickle.load(open(result_path, "rb"))["seqs"]
            best_model = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), "rb")
            )["order"][0]
            iptm_ptm_score = json.load(
                open(os.path.join(result_subdir, "ranking_debug.json"), "rb")
            )["iptm+ptm"][best_model]
            check_dict = pickle.load(
                open(os.path.join(result_subdir, f"result_{best_model}.pkl"), "rb")
            )
            iptm_score = check_dict["iptm"]
            pae_mtx = check_dict["predicted_aligned_error"]
            check = examine_inter_pae(pae_mtx, seqs, cutoff=FLAGS.cutoff)
            mpDockq_score = obtain_mpdockq(os.path.join(FLAGS.output_dir, job))
            if check:
                good_jobs.append(str(job))
                iptm_ptm.append(iptm_ptm_score)
                iptm.append(iptm_score)
                mpDockq_scores.append(mpDockq_score)

            logging.info(f"done for {job} {count} out of {len(jobs)} finished.")

    pi_score_df = run_and_summarise_pi_score(
        FLAGS.output_dir, good_jobs, FLAGS.surface_thres
    )
    pi_score_df["iptm+ptm"] = iptm_ptm
    pi_score_df["mpDockQ"] = mpDockq_scores
    pi_score_df["iptm"] = iptm
    columns = list(pi_score_df.columns.values)
    columns.pop(columns.index("jobs"))
    pi_score_df = pi_score_df[["jobs"] + columns]
    pi_score_df = pi_score_df.sort_values(by="iptm", ascending=False)
    pi_score_df.to_csv(
        os.path.join(FLAGS.output_dir, "predictions_with_good_interpae.csv"),
        index=False,
    )

    if FLAGS.create_notebook:
        create_notebook(pi_score_df, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
