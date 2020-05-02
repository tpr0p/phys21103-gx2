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

# data and save paths - round 1
ROUND1_NA22_CALIBRATION_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                      "round1_Na22_calibration.tsv")
ROUND1_NA22_CALIBRATION_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                      "round1_na22_calibration_spectrum.png")        
ROUND1_NA22_BACKGROUND_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                     "round1_Na22_void.tsv")
ROUND1_NA22_BACKGROUND_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                     "round1_na22_background_sepctrum.png")
ROUND1_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5",
]
ROUND1_NA22_STEP_COUNT = len(ROUND1_NA22_STEP_NAMES)
ROUND1_NA22_STEP_DATA_FILE_PATHS = list()
ROUND1_NA22_STEP_SAVE_FILE_PATHS = list()
for step_name in ROUND1_NA22_STEP_NAMES:
    ROUND1_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round1_Na22_{}.tsv".format(step_name)))
    ROUND1_NA22_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                         "round1_na22_{}_spectrum.png".format(step_name)))
#ENDFOR

# data and save paths - round 2
ROUND2_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5",
]
ROUND2_NA22_STEP_COUNT = len(ROUND2_NA22_STEP_NAMES)
ROUND2_NA22_STEP_DATA_FILE_PATHS = list()
ROUND2_NA22_STEP_SAVE_FILE_PATHS = list()
for step_name in ROUND2_NA22_STEP_NAMES:
    ROUND2_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round2_na22_{}.tsv".format(step_name)))
    ROUND2_NA22_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                         "round2_na22_{}_spectrum.png".format(step_name)))
#ENDFOR

# data and save paths - round 3
ROUND3_NA22_CALIBRATION_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                      "round3_na22_calibration.tsv")
ROUND3_NA22_CALIBRATION_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                      "round3_na22_calibration_spectrum.png")
ROUND3_NA22_CALIBRATION2_DATA_FILE_PATH = os.path.join(DATA_PATH,
                                                      "round3_na22_calibration2.tsv")
ROUND3_NA22_CALIBRATION2_SAVE_FILE_PATH = os.path.join(ASSETS_PATH,
                                                      "round3_na22_calibration2_spectrum.png")        
ROUND3_NA22_STEP_NAMES = [
    # "0.5mm_bottom",
    # "6mm_onefourth",
    # "6mm_Mid", "4.5mm_Mid", "3mm_Mid", "2mm_Mid", "1.5mm_Mid", "1mm_Mid", "0.5mm_Mid",
    # "6mm_threefourths",
    "6mm_top", "4.5mm_top", "3mm_top", "2mm_top", "1.5mm_top", "1mm_top", "0.5mm_top",
]
ROUND3_NA22_STEP_COUNT = len(ROUND3_NA22_STEP_NAMES)
ROUND3_NA22_STEP_DATA_FILE_PATHS = list()
ROUND3_NA22_STEP_SAVE_FILE_PATHS = list()
for step_name in ROUND3_NA22_STEP_NAMES:
    ROUND3_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round3_na22_{}.tsv".format(step_name)))
    ROUND3_NA22_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                         "round3_na22_{}_spectrum.png".format(step_name)))
#ENDFOR
    

# vlines - round 0
# eyeball
ROUND0_NA22_CALIBRATION_VLINES = np.array([730, 820]) # 83 / 5 background
# taken from fit_round0_na22_127.py and adjusted
ROUND0_NA22_STEP_VLINES = np.array([
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
    [730, 820],
])
# ROUND0_NA22_STEP_VLINES = np.array([
#     
#     [725, 825],
#     [720, 820],
#     [720, 820],
#     [720, 820],
#     [725, 820],
#     [720, 820],
#     [720, 825],
#     [720, 820],
# ])

# vlines - round 1
ROUND1_NA22_CALIBRATION_VLINES = np.array([700, 800])
ROUND1_NA22_STEP_VLINES = np.array([
    [670, 770],
    [670, 770],
    [670, 770],
    [670, 770],
    [670, 770],
])

# vlines - round 2
ROUND2_NA22_CALIBRATION_VLINES = np.array([700, 800])
ROUND2_NA22_STEP_VLINES = np.array([
    [670, 770],
    [670, 770],
    [670, 770],
    [670, 770],
    [680, 780],
])

# constants
TAB_DELIMITER = "\t"
DPI = int(1e3)
ROUND0_SKIPROW = 23
ROUND1_SKIPROW = 24
ROUND2_SKIPROW = 24
ROUND3_SKIPROW = 24
COLS = (0, 2)
COUNT_DTYPE = int


def plot_spectrum(data_path, save_path,
                  title=None,
                  vlines=None,
                  skiprows=0):
    """
    Plot the spectrum at `data_path` and save the plot to `save_path`.

    Arguments:
    data_path :: str - The file path to a .tsv file.
    save_path :: str - The file path to a .png file.
    title :: str - The plot title.
    vlines :: ndarray(vline_count) - vline coordintates
    
    Returns: None
    """
    data = np.loadtxt(data_path,
                      delimiter=TAB_DELIMITER,
                      dtype=COUNT_DTYPE,
                      skiprows=skiprows, usecols=COLS,)
    channels = data[:, 0]
    counts = data[:, 1]
    
    plt.figure()
    plt.plot(channels, counts)
    if title is not None:
        plt.title(data_path)
    #ENDIF
    if vlines is not None:
        for vline in vlines:
            plt.axvline(x=vline, color="black")
        #ENDFOR
        if vlines.shape[0] == 2:
            gross_counts = np.sum(counts[vlines[0]:vlines[1] + 1])
            text_x = vlines[1] + 10
            text_y = np.amax(counts) * .8
            text_str = "G = {}".format(gross_counts)
            plt.text(text_x, text_y, text_str)
        #ENDIF
    #ENDIF
    plt.ylabel("Counts")
    plt.xlabel("Channel")
    plt.ylim(ymin=0)
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
                      ROUND0_NA22_CALIBRATION_SAVE_FILE_PATH,
                      vlines=ROUND0_NA22_CALIBRATION_VLINES,
                      skiprows=ROUND0_SKIPROW)
    # Round 0 - Na22 Background
    if False:
        plot_spectrum(ROUND0_NA22_BACKGROUND_DATA_FILE_PATH,
                      ROUND0_NA22_BACKGROUND_SAVE_FILE_PATH,
                      skiprows=ROUND0_SKIPROW)
    # Round 0 - Na22 Steps
    if False:
        indices = range(ROUND0_NA22_STEP_COUNT)
        # indices = [0]
        for i in indices:
            plot_spectrum(ROUND0_NA22_STEP_DATA_FILE_PATHS[i],
                          ROUND0_NA22_STEP_SAVE_FILE_PATHS[i],
                          vlines=ROUND0_NA22_STEP_VLINES[i],
                          skiprows=ROUND0_SKIPROW)
    #ENDIF

    # Round 1 - Na22 Calibration
    if False:
        plot_spectrum(ROUND1_NA22_CALIBRATION_DATA_FILE_PATH,
                      ROUND1_NA22_CALIBRATION_SAVE_FILE_PATH,
                      vlines=ROUND1_NA22_CALIBRATION_VLINES,
                      skiprows=ROUND1_SKIPROW)
    # Round 1 - Na22 steps
    if False:
        indices = range(ROUND1_NA22_STEP_COUNT)
        # indices = [0]
        for i in indices:
            plot_spectrum(ROUND1_NA22_STEP_DATA_FILE_PATHS[i],
                          ROUND1_NA22_STEP_SAVE_FILE_PATHS[i],
                          vlines=ROUND1_NA22_STEP_VLINES[i],
                          skiprows=ROUND1_SKIPROW)
    #ENDIF

    # Round 2 - Na22 steps
    if False:
        # indices = range(ROUND2_NA22_STEP_COUNT)
        indices = [4]
        for i in indices:
            plot_spectrum(ROUND2_NA22_STEP_DATA_FILE_PATHS[i],
                          ROUND2_NA22_STEP_SAVE_FILE_PATHS[i],
                          vlines=ROUND2_NA22_STEP_VLINES[i],
                          skiprows=ROUND2_SKIPROW)
    #ENDIF
    
    # Round 3 - Na22 Calibration
    if False:
        plot_spectrum(ROUND3_NA22_CALIBRATION_DATA_FILE_PATH,
                      ROUND3_NA22_CALIBRATION_SAVE_FILE_PATH,
                      skiprows=ROUND3_SKIPROW)
        
    # Round 3 - Na22 Calibration 2
    if False:
        plot_spectrum(ROUND3_NA22_CALIBRATION2_DATA_FILE_PATH,
                      ROUND3_NA22_CALIBRATION2_SAVE_FILE_PATH,
                      skiprows=ROUND3_SKIPROW)
        
    # Round 3 - Na22 steps
    if False:
        # indices = range(ROUND3_NA22_STEP_COUNT)
        indices = [0]
        for i in indices:
            plot_spectrum(ROUND3_NA22_STEP_DATA_FILE_PATHS[i],
                          ROUND3_NA22_STEP_SAVE_FILE_PATHS[i],
                          skiprows=ROUND3_SKIPROW)
    #ENDIF
#ENDDEF


if __name__ == "__main__":
    main()
