import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def plot_pae_from_matrix(seqs, pae_matrix, figure_name='', ranking: int = 0):
    xticks = []
    xticks_labels = []
    initial_tick = 0

    for s in seqs:
        # Label the start of the sequence
        if initial_tick == 0:
            xticks.append(initial_tick)
            xticks_labels.append(str(initial_tick + 1))

        initial_tick += len(s)

        # Label the end of the sequence
        xticks.append(initial_tick - 1)
        xticks_labels.append(str(initial_tick))

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    pos = ax1.imshow(pae_matrix, cmap="bwr", vmin=0, vmax=30)

    ax1.set_xticks(xticks)
    ax1.set_yticks(xticks)

    ax1.set_xticklabels(xticks_labels, size="large", rotation=45, ha="right")
    ax1.set_yticklabels(xticks_labels, size="large")

    fig.colorbar(pos).ax.set_title("unit: Angstrom")

    # Draw black lines at sequence boundaries
    for t in range(1, len(xticks)):  # Draw lines at the end of each sequence
        ax1.axhline(xticks[t], color="black", linewidth=3.5)
        ax1.axvline(xticks[t], color="black", linewidth=3.5)

    plt.title(f"ranked_{ranking}")
    plt.savefig(figure_name)
    plt.close(fig)
