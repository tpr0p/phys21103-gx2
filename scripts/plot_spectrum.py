"""
plot_spectrum.py - This module will be used for plotting spectrums.

Author: Thomas
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Construct paths.
if not "PHYS_GX2_PATH" in os.environ:
    WDIR = "."
else:
    WDIR = os.environ["PHYS_GX2_PATH"]
DATA_PATH = os.path.join(WDIR, "data")
ASSETS_PATH = os.path.join(WDIR, "assets")

# data and save paths
ROUND0_NA22_CALIBRATION_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                      "round0_CalibrationNa22Gain3Time60s.tsv")
ROUND0_NA22_CALIBRATION_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                      "round0_na22_calibration_spectrum.png")        
ROUND0_NA22_BACKGROUND_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                     "round0_Na22_void.tsv")
ROUND0_NA22_BACKGROUND_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                     "round0_na22_background_sepctrum.png")
ROUND0_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"
]
ROUND0_NA22_STEP_COUNT = len(ROUND0_NA22_STEP_NAMES)
ROUND0_NA22_STEP_DATA_FILE_PATHS = list()
ROUND0_NA22_STEP_SAVE_FILE_PATHS = list()
for step_name in ROUND0_NA22_STEP_NAMES:
    ROUND0_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round0_Na22_{}.tsv".format(step_name)))
    ROUND0_NA22_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                         "round0_na22_{}_spectrum.png".format(step_name)))
#ENDFOR


# constants
TAB_DELIMITER = "\t"
DPI = int(1e3)
SKIPROW = 23
COLS = (0, 2)
COUNT_DTYPE = int


def plot_spectrum(data_path, save_path,
                  title=None):
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
    plt.plot(channels, counts)
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
        plot_spectrum(ROUND0_NA22_CALIBRATION_DATA_FILE_PATH,
                      ROUND0_NA22_CALIBRATION_SAVE_FILE_PATH)
    # Round 0 - Na22 Background
    if False:
        plot_spectrum(ROUND0_NA22_BACKGROUND_DATA_FILE_PATH,
                      ROUND0_NA22_BACKGROUND_SAVE_FILE_PATH)
    # Round 0 - Na22 Steps
    if False:
        for i in range(ROUND0_NA22_STEP_COUNT):
            plot_spectrum(ROUND0_NA22_STEP_DATA_FILE_PATHS[i],
                          ROUND0_NA22_STEP_SAVE_FILE_PATHS[i])
    #ENDIF

    return
#ENDDEF


if __name__ == "__main__":
    main()
