import argparse
import io
from alphapulldown.utils.create_combinations import process_files
from alphapulldown.utils.modelling_setup import parse_fold, create_custom_info, create_interactors
from alphapulldown.objects import MultimericObject
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def create_multimer_objects(args):
    data = create_custom_info(args.parsed_input)
    interactors = create_interactors(data, args.features_directory, 0)
    multimer = MultimericObject(interactors[0])
    return multimer


def profile_all_jobs_and_cluster(all_folds: List[str], args):
    output = {"name": [],
              "msa_depth": [],
              "seq_length": []}
    total_num = len(all_folds)
    for idx, i in enumerate(all_folds):
        args.input = [i]
        args = parse_fold(args)
        multimer = create_multimer_objects(args)
        msa_depth, seq_length = multimer.feature_dict["msa"].shape
        output['name'].append(i)
        output['msa_depth'].append(msa_depth)
        output['seq_length'].append(seq_length)
        progress = ((idx+1)*100)/total_num
        if (idx + 1) % 10 ==0:
            logger.info(f"Finished profiling {idx +1} out of {total_num} jobs. {progress:.1f}% completed")
    return pd.DataFrame.from_dict(output)


def plot_clustering_result(X : np.array, labels : List[float | int], num_cluster: int,
                        output_dir: str) -> None:
    total_num = len(labels)
    labels_unique = np.unique(labels)
    n_clusters_ = num_cluster
    cmap = plt.cm.get_cmap('tab20')

    norm = plt.Normalize(vmin=min(labels_unique), vmax=max(labels_unique))
    color_template = {label: cmap(norm(label)) for label in labels_unique}
    for label in labels_unique:
        my_members = labels == label
        col = color_template[label]
        plt.scatter(X[my_members, 0], X[my_members, 1], color=col)
    plt.xlabel('seq_length')
    plt.ylabel("msa_depth")
    plt.title(
        f"Total number of jobs: {total_num} Estimated number of clusters: {n_clusters_}")
    plt.savefig(f"{output_dir}/clustered_prediction_jobs.png")

def write_individual_job_cluster(all_jobs : pd.DataFrame,
                                 labels : list, output_dir : str) -> None:
    unique_labels = np.unique(labels)
    X = all_jobs.loc[:, ['seq_length', 'msa_depth']].values
    for i, label in enumerate(unique_labels):
        is_member = labels == label
        max_seq_length, max_msa_depth = max(X[is_member, 0]), max(X[is_member, 1])
        members = all_jobs[is_member]['name'].tolist()
        file_name = f"{output_dir}/job_cluster{i+1}_{int(max_seq_length)}_{int(max_msa_depth)}.txt"
        with open(file_name, "w") as outfile:
            for m in members:
                print(m, file=outfile)
            outfile.close()

def cluster_jobs(all_folds, args):
    all_jobs = profile_all_jobs_and_cluster(all_folds, args)
    seq_lengths = all_jobs['seq_length'].values
    max_diff = 150 
    num_cluster = int((np.max(seq_lengths) - np.min(seq_lengths)) / max_diff) + 1
    # Assign elements to bins
    labels = []
    for value in seq_lengths:
        bin_index = int((value - np.min(seq_lengths)) // max_diff)
        labels.append(bin_index)

    write_individual_job_cluster(all_jobs, labels, args.output_dir)
    X = all_jobs.loc[:, ['seq_length', 'msa_depth']].values
    plot_clustering_result(X, labels, num_cluster,args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run protein folding.")
    parser.add_argument(
        "--protein_lists",
        dest="protein_lists",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="protein list files"
    )
    parser.add_argument(
        "--protein_delimiter",
        dest="protein_delimiter",
        type=str,
        default="+",
        required=False,
        help="protein list files"
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="pulldown",
        required=True,
        help="protein list files"
    )
    parser.add_argument(
        "--features_directory",
        dest="features_directory",
        type=str,
        nargs="+",
        required=True,
        help="Path to computed monomer features.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./",
        required=False,
        help="output directory"
    )
    args = parser.parse_args()
    protein_lists = args.protein_lists
    if args.mode == "all_vs_all":
        protein_lists = [args.protein_lists[0], args.protein_lists[0]]
    # buffer = io.StringIO()
    import time
    start = time.time()
    all_combinations = process_files(input_files=protein_lists)

    all_folds = ["+".join(combo) for combo in all_combinations]
    all_folds = [x.strip().replace(",", ":") for x in all_folds]
    all_folds = [x.strip().replace(";", "+") for x in all_folds]
    end = time.time()
    diff1 = end - start 
    cluster_jobs(all_folds, args)
    end = time.time()
    diff2 = end - start 
    logger.info(f"process_files steps takes {diff1}s and total time is: {diff2}")


if __name__ == "__main__":
    main()
