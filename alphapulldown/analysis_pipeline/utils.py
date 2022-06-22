import IPython.display as display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from programme_notebook.af2hyde_mod import plot_predicted_alignment_error


def display_pae_plots(subdir):
    """A function to display all the pae plots in the subdir"""
    images = sorted([i for i in os.listdir(subdir) if ".png" in i])
    if len(images) > 0:
        fig, axs = plt.subplots(1, len(images), figsize=(500, 500))
        for i in range(len(images)):
            img = plt.imread(os.path.join(subdir, images[i]))
            axs[i].imshow(img)
            axs[i].axis("off")
        plt.show()
    else:
        plot_predicted_alignment_error(subdir)
