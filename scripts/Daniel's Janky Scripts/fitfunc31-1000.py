"""
fit_round0_na22_127.py - This module contains guassian fits to the 1.27 MeV
full energy peak for the Na22 source using round 0 data.

Author: Thomas
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Construct paths.
if not "Remote_Gamma_Cross" in os.environ:
    WDIR = "."
else:
    WDIR = os.environ["Remote_Gamma_Cross"]
#ENDIF
DATA_PATH = os.path.join(WDIR, "data")
ASSETS_PATH = os.path.join(WDIR, "assets")

# data and save paths
ROUND0_BA133_STEP_NAMES = [
    "h1", "h3",  "h5", "h7", "h9"
]
ROUND0_STEP_COUNT = len(ROUND0_BA133_STEP_NAMES)
ROUND0_BA133_STEP_DATA_FILE_PATHS = list()
ROUND0_BA133_81_STEP_SAVE_FILE_PATHS = list()
ROUND0_BA133_81_STEP_PLOT_TITLES = list()
for step_name in ROUND0_BA133_STEP_NAMES:
    ROUND0_BA133_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "BA133_{}_1000s.tsv".format(step_name)))
    ROUND0_BA133_81_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                             "BA133_31_{}_1000s_fit.png".format(step_name)))
    ROUND0_BA133_81_STEP_PLOT_TITLES.append("Round 0 - BA133 - 1000s - 31Kev - {}".format(step_name.upper()))
#ENDFOR
ROUND0_BA133_81_STEP_NET_COUNTS_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round0_BA133_31_1000_step_netcounts.png"
)


# data constants
ROUND0_BA133_81_STEP_INITIAL_PARAMETERS = [700., 58.,7., 1., 1.]
ROUND0_BA133_81_H1_CHANNEL_LO = 40
ROUND0_BA133_81_H1_CHANNEL_HI = 82

ROUND0_BA133_81_H3_CHANNEL_LO = 40
ROUND0_BA133_81_H3_CHANNEL_HI = 82

ROUND0_BA133_81_H5_CHANNEL_LO = 40
ROUND0_BA133_81_H5_CHANNEL_HI = 82

ROUND0_BA133_81_H7_CHANNEL_LO = 40
ROUND0_BA133_81_H7_CHANNEL_HI = 82

ROUND0_BA133_81_H9_CHANNEL_LO = 40
ROUND0_BA133_81_H9_CHANNEL_HI = 82
ROUND0_BA133_81_H1_PLOT_TEXT = (
    2,
    150,
    "$N = 2979 \pm 149$ counts\n"
    "$\mu = 58 \pm 1$ channel\n"
    "$\sigma = 6.9 \pm 0.3$ channel\n"
    "$A = 23 \pm 10$ counts\n"
    "$B = (4.4 \pm 1.3) 10^{-1}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.11$\n",
)

ROUND0_BA133_81_H3_PLOT_TEXT = (
    3,
    150,
    "$N = 2619 \pm 141$ counts\n"
    "$\mu = 58 \pm 1$ channel\n"
    "$\sigma = 7.0 \pm 0.3$ channel\n"
    "$A = 16 \pm 10$ counts\n"
    "$B = (5 \pm 1.3) 10^{-1}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.86$\n",
)

ROUND0_BA133_81_H5_PLOT_TEXT = (
    1,
    150,
    "$N = 2101 \pm 132$ counts\n"
    "$\mu = 58 \pm 1$ channel\n"
    "$\sigma = 6.8 \pm 0.3$ channel\n"
    "$A = 10 \pm 9$ counts\n"
    "$B = (6.3 \pm 1.3) 10^{-1}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.00$\n",
)

ROUND0_BA133_81_H7_PLOT_TEXT = (
    1,
    70,
    "$N = 1765 \pm 160$ counts\n"
    "$\mu = 58 \pm 1$ channel\n"
    "$\sigma = 7.4 \pm 0.5$ channel\n"
    "$A = 10 \pm 11$ counts\n"
    "$B = (5.7 \pm 1.4) 10^{-1}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.05$\n",
)

ROUND0_BA133_81_H9_PLOT_TEXT = (
    80,
    15,
    "$N = 826 \pm 135$ counts\n"
    "$\mu = 58 \pm 1$ channel\n"
    "$\sigma = 7.2 \pm 0.9$ channel\n"
    "$A = 17 \pm 9$ counts\n"
    "$B = (4.76 \pm 1.2) 10^{-1}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.01$\n",
)
ROUND0_BA133_81_STEP_DATA = [
    (
        ROUND0_BA133_81_H1_CHANNEL_LO,
        ROUND0_BA133_81_H1_CHANNEL_HI,
        ROUND0_BA133_81_H1_PLOT_TEXT,
    ),
  

        (
        ROUND0_BA133_81_H3_CHANNEL_LO,
        ROUND0_BA133_81_H3_CHANNEL_HI,
        ROUND0_BA133_81_H3_PLOT_TEXT,
    ),
        
        (
        ROUND0_BA133_81_H5_CHANNEL_LO,
        ROUND0_BA133_81_H5_CHANNEL_HI,
        ROUND0_BA133_81_H5_PLOT_TEXT,
    ),
        
        (
        ROUND0_BA133_81_H7_CHANNEL_LO,
        ROUND0_BA133_81_H7_CHANNEL_HI,
        ROUND0_BA133_81_H7_PLOT_TEXT,
    ),
        
        (
        ROUND0_BA133_81_H9_CHANNEL_LO,
        ROUND0_BA133_81_H9_CHANNEL_HI,
        ROUND0_BA133_81_H9_PLOT_TEXT,
    ),
]
ROUND0_BA133_81_STEP_NET_COUNTS = np.array([
    (2979, 149),
    (2619, 141),
    (2101, 132),
    (1765, 160),
    (826, 135)
])


# processing constants
TAB_DELIMITER = "\t"
DPI = int(1e3)
SKIPROW = 25
COLS = (0, 2)
COUNT_DTYPE = int
FIT_PARAMETER_COUNT = 5
PLOT_FIT_SAMPLES = int(1e3)
PLOT_CHANNEL_BUF = 40
PLOT_TEXT_FONT_SIZE = 8
MARKER_SIZE = 5


def fitfunc(p, x):
    """
    Fit to a Guassian with linear background:
    \frac{N_{0}}{\sigma 2 \pi} \exp{\frac{-(x - \mu)^{2}}{2 {\sigma}^{2}}} + A + Bx
    """
    return (
        (p[0] / (p[2] * np.sqrt(2 * np.pi))) * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))
            + p[3] + p[4] * x
    )


def residual(p, x, y, err):
    """
    Use a simple residual that incorporates y error.
    """
    yp = fitfunc(p, x)
    return (yp - y) / err


def fit_and_plot(data_file_path,
                 channel_lo=0, channel_hi=1000,
                 initial_parameters=None,
                 plot_title=None,
                 plot_text=None,
                 save_file_path=None):
    """
    Perform a guassian fit to the specified data and the specified region.
    
    Arguments:
    channel_hi :: int - The last channel of the fit interval.
    channel_lo :: int - The first channel of the fit interval.
    data_file_path :: str - The path to a .tsv file containing the count data.
    initial_parameters :: ndarray(FIT_PARAMETER_COUNT) - An initial guess
        for the fit parameters.
    plot_title :: str - The title for the plot.
    plot_text :: tuple(int, int, str) - The text to place on the plot and the
        x-y coordinates of its location.
    save_file_path :: str - The path to a .png file to save the fit plot to.
        If none is specified then the no fitting occurs.
    
    Returns: None
    """
    if initial_parameters is None:
        initial_parameters = np.ones(FIT_PARAMETER_COUNT, dtype=np.float64)

    # Fit the data.
    data = np.loadtxt(
        data_file_path,
        delimiter=TAB_DELIMITER,
        skiprows=SKIPROW,
        usecols=COLS,
    )
    channels = data[:, 0]
    counts = data[:, 1]
    dcounts = np.sqrt(counts)
    # Replace 0 uncertainty with 1 uncertainty.
    # Note that dN = sqrt(N) is a bad approximation for small counts.
    dcounts = np.where(dcounts, dcounts, 1.)
    channels_slice = channels[channel_lo : channel_hi + 1]
    counts_slice = counts[channel_lo : channel_hi + 1]
    dcounts_slice = dcounts[channel_lo : channel_hi + 1]
    pf, cov, info, mesg, success = optimize.leastsq(
        residual, initial_parameters, args=(channels_slice, counts_slice, dcounts_slice),
        full_output=1,
    )

    chisq = sum(info["fvec"] ** 2)
    dof = len(channels_slice) - len(pf)
    rchisq = chisq/dof
    pferr = [np.sqrt(cov[i, i]) for i in range(len(pf))]
    
    print("p:\n{}".format(pf))
    print("dp:\n{}".format(pferr))
    print("rchisq:\n{}".format(rchisq))

    # Plot the fit.
    if save_file_path is not None:
        channel_extend_lo = np.maximum(0, channel_lo - PLOT_CHANNEL_BUF)
        channel_extend_hi = np.minimum(1000, channel_hi + PLOT_CHANNEL_BUF)
        channels_extend = np.hstack((np.arange(channel_extend_lo, channel_lo),
                                     np.arange(channel_hi + 1, channel_extend_hi + 1)))
        counts_extend = counts[channels_extend]
        channels_fit = np.linspace(channel_lo, channel_hi, num=PLOT_FIT_SAMPLES)
        counts_fit = fitfunc(pf, channels_fit)
        plt.figure()
        plt.scatter(channels_extend, counts_extend, label="data", color="blue", s=MARKER_SIZE)
        plt.errorbar(channels_slice, counts_slice, yerr=dcounts_slice,
                     fmt=".", ms=MARKER_SIZE, linestyle="None", label="data in fit", color="red")
        plt.plot(channels_fit, counts_fit, label="fit", color="black")
        if plot_title is not None:
            plt.title(plot_title)
        plt.ylabel("Counts")
        plt.xlabel("Channel")
        plt.legend()
        if plot_text is not None:
            plt.text(*plot_text, fontdict={"size": PLOT_TEXT_FONT_SIZE})
        plt.savefig(save_file_path, dpi=DPI)
    #ENDIF

    return


def plot_ba133_81_step_counts():
    """
    Plot the net count obtained from the Na22 1.27 MeV peak fit for each
    step measurement.
    
    Arguments: None
    Returns: None
    """
    plt.figure()
    net_counts = ROUND0_BA133_81_STEP_NET_COUNTS[:, 0]
    dnet_counts = ROUND0_BA133_81_STEP_NET_COUNTS[:, 1]
    step_axis = [1,3,5,7,9]
    plt.errorbar(step_axis, net_counts, yerr=dnet_counts, linestyle="None",
                 fmt=".", ms=MARKER_SIZE)
    plt.title("Ba22 - 31 keV 1000s - Height Step Net Counts")
    plt.ylabel("Net Count")
    plt.xlabel("Height Step")
    plt.savefig(ROUND0_BA133_81_STEP_NET_COUNTS_SAVE_FILE_PATH,
                dpi=DPI)
    return


def main():
    """
    Do all the things.
    """
    # Round 0 - Na22 - 1.27 MeV - Height Steps
    if 1==1:
        # all
        round0_ba133_81_step_indices = np.arange(ROUND0_STEP_COUNT)
        # selected
        # round0_na22_127_step_indices = np.array([
        #     2,
        # ])
        for i in round0_ba133_81_step_indices:
            (channel_lo, channel_hi, plot_text) = ROUND0_BA133_81_STEP_DATA[i]
            fit_and_plot(
                ROUND0_BA133_STEP_DATA_FILE_PATHS[i],
                channel_hi=channel_hi,
                channel_lo=channel_lo,
                initial_parameters=ROUND0_BA133_81_STEP_INITIAL_PARAMETERS,
                plot_text=plot_text,
                plot_title=ROUND0_BA133_81_STEP_PLOT_TITLES[i],
                save_file_path=ROUND0_BA133_81_STEP_SAVE_FILE_PATHS[i],
            )
        #ENDFOR
    #ENDIF
    if 1==1:
        plot_ba133_81_step_counts()
    #ENDIF




if __name__ == "__main__":
    main()
