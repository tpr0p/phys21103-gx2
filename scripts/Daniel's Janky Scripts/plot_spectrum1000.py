"""
plot_spectrum.py - This module will be used for plotting spectrums.

Author: Thomas
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Construct paths.
if not "Remote_Gamma_Cross" in os.environ:
    WDIR = "."
else:
    WDIR = os.environ["Remote_Gamma_Cross"]
DATA_PATH = os.path.join(WDIR, "data")
ASSETS_PATH = os.path.join(WDIR, "assets")

# data and save paths
ROUND0_BA133_CALIBRATION_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                      "CallibrationBa133Gain10x.tsv")
ROUND0_BA133_CALIBRATION_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                      "Ba_133_calibration_spectrum.png")        
ROUND0_BA133_BACKGROUND_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                     "Ba133_null.tsv")
ROUND0_BA133_BACKGROUND_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                     "Ba_133_background_sepctrum.png")
ROUND0_BA133_STEP_NAMES = [
    "h1", "h3",  "h5", "h7","h9"
]
ROUND0_BA133_STEP_COUNT = len(ROUND0_BA133_STEP_NAMES)
ROUND0_BA133_STEP_DATA_FILE_PATHS = list()
ROUND0_BA133_STEP_SAVE_FILE_PATHS = list()
for step_name in ROUND0_BA133_STEP_NAMES:
    ROUND0_BA133_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "Ba133_{}_1000s.tsv".format(step_name)))
    ROUND0_BA133_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                         "BA133_{}_1000s_spectrum.png".format(step_name)))
#ENDFOR


# constants
TAB_DELIMITER = "\t"
DPI = int(1e3)
SKIPROW = 25
COLS = (0, 2)
COUNT_DTYPE = int


def plot_spectrum(data_path, save_path,
                  title='k'):
    """
    Plot the spectrum at `data_path` and save the plot to `save_path`.

    Arguments:
    data_path :: str - The file path to a .tsv file.
    save_path :: str - The file path to a .png file.
    title :: str - The plot title.
    
    Returns: None
    """

    data = np.loadtxt(data_path,
                      delimiter=TAB_DELIMITER,
                      dtype=COUNT_DTYPE,
                      skiprows=SKIPROW, usecols=COLS,)
    channels = data[:, 0]
    counts = data[:, 1]
    
    plt.figure()
    plt.plot(channels, counts,"r.")
    if title is not None:
        plt.title(data_path)
    plt.ylabel("Counts")
    plt.xlabel("Channel")
    plt.savefig(save_path, dpi=DPI)
    
    return
#ENDDEF


def main():
    """
    Plot all the things.
    """
    # Round 0 - Na22 Calibration
    if False:
    	plot_spectrum(ROUND0_BA133_CALIBRATION_DATA_FILE_PATH,
                      ROUND0_BA133_CALIBRATION_SAVE_FILE_PATH)
    # Round 0 - Na22 Background
    if False:
    	plot_spectrum(ROUND0_BA133_BACKGROUND_DATA_FILE_PATH,
                      ROUND0_BA133_BACKGROUND_SAVE_FILE_PATH)
    # Round 0 - Na22 Steps
    
    for i in range(ROUND0_BA133_STEP_COUNT):
        plot_spectrum(ROUND0_BA133_STEP_DATA_FILE_PATHS[i],
                          ROUND0_BA133_STEP_SAVE_FILE_PATHS[i])
    #ENDIF

    return
#ENDDEF


if __name__ == "__main__":
    main()
