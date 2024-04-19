#
# the script to plot PAEs after predicting structures
# #
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def plot_pae_from_matrix(seqs,pae_matrix,figure_name='',ranking:int = 0):
    xticks = []
    initial_tick = 0
    for s in seqs:
        initial_tick = initial_tick + len(s)
        xticks.append(initial_tick)

    xticks_labels = []
    for i, t in enumerate(xticks):
        xticks_labels.append(str(i + 1))

    yticks_labels = []
    for s in seqs:
        yticks_labels.append(str(len(s)))
    fig, ax1 = plt.subplots(1, 1)
    # plt.figure(figsize=(3,18))
    check = pae_matrix
    fig, ax1 = plt.subplots(1, 1)
    pos = ax1.imshow(check, cmap="bwr", vmin=0, vmax=30)
    ax1.set_xticks(xticks)
    ax1.set_yticks(xticks)

    ax1.set_xticklabels(xticks_labels, size="large")
    ax1.set_yticklabels(yticks_labels,size="large")
    fig.colorbar(pos).ax.set_title("unit: Angstrom")
    for t in xticks:
        ax1.axhline(t, color="black", linewidth=3.5)
        ax1.axvline(t, color="black", linewidth=3.5)
    plt.title(f"ranked_{ranking}".format(i))
    plt.savefig(figure_name)