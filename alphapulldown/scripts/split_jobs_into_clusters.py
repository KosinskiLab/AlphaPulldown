import argparse
import io
from alphapulldown.utils.create_combinations import process_files
from alphapulldown.utils.modelling_setup import parse_fold, create_custom_info, create_interactors
from alphapulldown.objects import MultimericObject
import pandas as pd
from typing import List
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import numpy as np


def create_multimer_objects(args):
    data = create_custom_info(args.parsed_input)
    interactors = create_interactors(data, args.features_directory, 0)
    multimer = MultimericObject(interactors)
    return multimer


def profile_all_jobs_and_cluster(all_folds: List[str], args):
    output = {"name": [],
              "msa_depth": [],
              "seq_length": []}
    for i in all_folds:
        args.input = i
        args = parse_fold(args)
        multimer = create_multimer_objects(args)
        msa_depth, seq_length = multimer.feature_dict["msa"].shape
        output['name'].append(i)
        output['msa_depth'].append(msa_depth)
        output['seq_length'].append(seq_length)
    return pd.DataFrame.from_dict(output)


def plot_clustering_result(X, labels, cluster_centers):
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
    plt.savefig('clustered_prediction_jobs.png')


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
    plot_clustering_result(X, labels, cluster_centres)


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
        "--features_directory",
        dest="features_directory",
        type=str,
        nargs="+",
        required=True,
        help="Path to computed monomer features.",
    )
    args = parser.parse_args()
    args.mode = "custom"
    protein_lists = args.protein_lists

    buffer = io.StringIO()
    _ = process_files(input_files=protein_lists, output_path=buffer)
    buffer.seek(0)
    all_folds = buffer.readlines()
    all_folds = [x.strip().replace(",", ":") for x in all_folds]
    # all_folds = [x.strip().replace(";", "+") for x in all_folds]
    cluster_jobs(all_folds, args)


if __name__ == "__main__":
    main()
