import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def plot_pae_from_matrix(seqs, pae_matrix, figure_name='', ranking: int = 0):
    xticks = []
    initial_tick = 0
    for s in seqs:
        initial_tick += len(s)
        xticks.append(initial_tick)

    xticks_labels = [str(i + 1) for i in range(len(xticks))]
    yticks_labels = [str(i + 1) for i in range(len(xticks))]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    pos = ax1.imshow(pae_matrix, cmap="bwr", vmin=0, vmax=30)

    ax1.set_xticks(xticks)
    ax1.set_yticks(xticks)

    ax1.set_xticklabels(xticks_labels, size="large")
    ax1.set_yticklabels(yticks_labels, size="large")

    ax1.set_xlim([min(xticks), max(xticks)])
    ax1.set_ylim([min(xticks), max(xticks)])

    fig.colorbar(pos).ax.set_title("unit: Angstrom")

    for t in xticks:
        ax1.axhline(t, color="black", linewidth=3.5)
        ax1.axvline(t, color="black", linewidth=3.5)

    plt.title(f"ranked_{ranking}")
    plt.savefig(figure_name)
    plt.close(fig)
