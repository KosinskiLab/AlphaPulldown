from configparser import Interpolation
import IPython.display as display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import pandas as pd
from absl import logging
import subprocess
import sys
#from analysis_pipeline.af2hyde_mod import plot_predicted_alignment_error
from af2plots.plotter import plotter


def display_pae_plots(subdir,figsize=(50, 50)):
    """A function to display all the pae plots in the subdir"""
    pattern = r"ranked_(\d+)\.png"
    images = sorted([i for i in os.listdir(subdir) if ".png" in i],
                    key= lambda x: int(re.search(pattern,x).group(1)))
    if len(images) > 0:
        fig, axs = plt.subplots(1, len(images), figsize=figsize)
        for i in range(len(images)):
            img = plt.imread(os.path.join(subdir, images[i]))
            axs[i].imshow(img,interpolation="nearest")
            axs[i].axis("off")
        #plt.show()
    else:
        #plot_predicted_alignment_error(subdir)
        af2o = plotter()
        dd = af2o.parse_model_pickles(subdir)
        ff = af2o.plot_predicted_alignment_error(dd)

    plt.show()