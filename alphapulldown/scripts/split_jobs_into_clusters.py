import argparse
import io
from alphapulldown.utils.create_combinations import process_files
from alphapulldown.utils.modelling_setup import parse_fold, create_custom_info, create_interactors
from alphapulldown.objects import MultimericObject
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

def create_multimer_objects(args):
    data = create_custom_info(args.parsed_input)
    interactors = create_interactors(data, args.features_directory, 0)
    multimer = MultimericObject(interactors)
    return multimer


def profile_all_jobs_and_cluster(all_folds: List[str], args):
    output = {"name": [],
              "msa_depth": [],
              "seq_length": []}
    total_num = len(all_folds)
    for idx, i in enumerate(all_folds):
        args.input = i
        args = parse_fold(args)
        multimer = create_multimer_objects(args)
        msa_depth, seq_length = multimer.feature_dict["msa"].shape
        output['name'].append(i)
        output['msa_depth'].append(msa_depth)
        output['seq_length'].append(seq_length)
        progress = ((idx+1)*100)/total_num
        if idx % 10 ==0:
            logger.info(f"Finished {idx +1} out of {total_num} {progress:3f} completed")
    return pd.DataFrame.from_dict(output)


def plot_clustering_result(X : np.array, labels : List[float | int], 
                           cluster_centers : list, output_dir: str) -> None:
    total_num = len(labels)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cmap = plt.cm.get_cmap('tab20')

    norm = plt.Normalize(vmin=min(labels_unique), vmax=max(labels_unique))
    color_template = {label: cmap(norm(label)) for label in labels_unique}
    for label in labels_unique:
        my_members = labels == label
        col = color_template[label]
        cluster_center = cluster_centers[label]
        plt.scatter(X[my_members, 0], X[my_members, 1], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
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
    print(f"start creating all jobs")
    all_jobs = profile_all_jobs_and_cluster(all_folds, args)
    X = all_jobs.loc[:, ['seq_length', 'msa_depth']].values
    estimated_bandwidth = estimate_bandwidth(X)
    if estimated_bandwidth == 0:
        estimated_bandwidth = 1.0
    cluster = MeanShift(bandwidth=estimated_bandwidth)
    result = cluster.fit(X)
    labels, cluster_centres = result.labels_, result.cluster_centers_
    write_individual_job_cluster(all_jobs, labels, args.output_dir)
    plot_clustering_result(X, labels, cluster_centres, args.output_dir)


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
