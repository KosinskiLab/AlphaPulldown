#
# the script to plot PAEs after predicting structures
# #
import pandas as pd
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def plot_pae(seqs: list, order, feature_dir, job_name):
    """
    a function to plot pae after all predictions are finished

    Arg
    seqs: a list of sub-unit's sequence
    order: ranking-debug.json 'order' value
    feature_dir: directory where the feature pickles are stored
    job_name: name of the job e.g. protein_A_and_protein_B
    """
    matplotlib.use("agg")
    outs = dict()
    for i in order:
        prediction_result = pkl.load(open(f"{feature_dir}/result_{i}.pkl", "rb"))
        outs[i] = prediction_result["predicted_aligned_error"]
        del prediction_result

    xticks = []
    initial_tick = 0
    for s in seqs:
        initial_tick = initial_tick + len(s)
        xticks.append(initial_tick)

    xticks_labels = []
    for i, t in enumerate(xticks):
        xticks_labels.append(str(i + 1))
    fig, ax1 = plt.subplots(1, 1)
    # plt.figure(figsize=(3,18))
    for i in range(len(order)):
        check = outs[order[i]]
        fig, ax1 = plt.subplots(1, 1)
        pos = ax1.imshow(check, cmap="bwr", vmin=0, vmax=30)
        ax1.set_xticks(xticks)
        ax1.set_yticks(xticks)

        ax1.set_xticklabels(xticks_labels, size="large")

        fig.colorbar(pos).ax.set_title("unit: Angstrom")
        for t in xticks:
            ax1.axhline(t, color="black", linewidth=3.5)
            ax1.axvline(t, color="black", linewidth=3.5)
        plt.title("ranked_{}".format(i))
        plt.savefig(f"{feature_dir}/{job_name}_PAE_plot_ranked_{i}.png")
