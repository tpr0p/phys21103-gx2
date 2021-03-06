"""
fit.py - This module contains gaussian fits to the 1.27 MeV and 511 keV
full energy peak for the Na22 source on varius data runs.

round 0 - Data Round 1
round 1 - Bonus Data 0.155cm absorber
round 2 - Bonus Data 0.598cm absorber
round 3 - Bonus Data

Author: Thomas
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Construct paths.
if not "PHYS_GX2_PATH" in os.environ:
    WDIR = "."
else:
    WDIR = os.environ["PHYS_GX2_PATH"]
#ENDIF
DATA_PATH = os.path.join(WDIR, "data")
ASSETS_PATH = os.path.join(WDIR, "assets")

# data and save paths
ROUND0_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"
]
ROUND0_STEP_COUNT = len(ROUND0_NA22_STEP_NAMES)
ROUND0_NA22_STEP_DATA_FILE_PATHS = list()
ROUND0_NA22_127_STEP_SAVE_FILE_PATHS = list()
ROUND0_NA22_127_STEP_PLOT_TITLES = list()
for step_name in ROUND0_NA22_STEP_NAMES:
    ROUND0_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round0_Na22_{}.tsv".format(step_name)))
    ROUND0_NA22_127_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                             "round0_na22_127_{}_fit.png".format(step_name)))
    ROUND0_NA22_127_STEP_PLOT_TITLES.append("Round 0 - Na22 - 1.27Mev - {}".format(step_name.upper()))
#ENDFOR
ROUND0_NA22_127_STEP_COUNTS_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round0_na22_127_step_counts.png"
)

ROUND1_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5",
]
ROUND1_STEP_COUNT = len(ROUND1_NA22_STEP_NAMES)
ROUND1_NA22_STEP_DATA_FILE_PATHS = list()
ROUND1_NA22_127_STEP_SAVE_FILE_PATHS = list()
ROUND1_NA22_127_STEP_PLOT_TITLES = list()
for step_name in ROUND1_NA22_STEP_NAMES:
    ROUND1_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round1_Na22_{}.tsv".format(step_name)))
    ROUND1_NA22_127_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                             "round1_na22_127_{}_fit.png".format(step_name)))
    ROUND1_NA22_127_STEP_PLOT_TITLES.append("Round 1 - Na22 - 1.27Mev - {}".format(step_name.upper()))
#ENDFOR
ROUND1_NA22_127_STEP_COUNTS_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round1_na22_127_step_counts.png"
)

ROUND2_NA22_STEP_NAMES = [
    "h1", "h2", "h3", "h4", "h5",
]
ROUND2_STEP_COUNT = len(ROUND2_NA22_STEP_NAMES)
ROUND2_NA22_STEP_DATA_FILE_PATHS = list()
ROUND2_NA22_127_STEP_SAVE_FILE_PATHS = list()
ROUND2_NA22_127_STEP_PLOT_TITLES = list()
for step_name in ROUND2_NA22_STEP_NAMES:
    ROUND2_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round2_na22_{}.tsv".format(step_name)))
    ROUND2_NA22_127_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                             "round2_na22_127_{}_fit.png".format(step_name)))
    ROUND2_NA22_127_STEP_PLOT_TITLES.append("Round 2 - Na22 - 1.27Mev - {}".format(step_name.upper()))
#ENDFOR
ROUND2_NA22_127_STEP_COUNTS_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round2_na22_127_step_counts.png"
)

ROUND3_NA22_STEP_NAMES = [
    "calibration",
    "6mm_Mid", "4.5mm_Mid", "3mm_Mid", "2mm_Mid", "1.5mm_Mid", "1mm_Mid", "0.5mm_Mid",
    "6mm_top", "4.5mm_top", "3mm_top", "2mm_top", "1.5mm_top", "1mm_top", "0.5mm_top"
]
ROUND3_STEP_COUNT = len(ROUND3_NA22_STEP_NAMES)
ROUND3_NA22_STEP_DATA_FILE_PATHS = list()
ROUND3_NA22_127_STEP_SAVE_FILE_PATHS = list()
ROUND3_NA22_127_STEP_PLOT_TITLES = list()
for step_name in ROUND3_NA22_STEP_NAMES:
    ROUND3_NA22_STEP_DATA_FILE_PATHS.append(os.path.join(DATA_PATH,
                                                         "round3_na22_{}.tsv".format(step_name)))
    ROUND3_NA22_127_STEP_SAVE_FILE_PATHS.append(os.path.join(ASSETS_PATH,
                                                             "round3_na22_127_{}_fit.png".format(step_name)))
    ROUND3_NA22_127_STEP_PLOT_TITLES.append("Round 3 - Na22 - 1.27Mev - {}".format(step_name.upper()))
#ENDFOR
ROUND3_NA22_127_STEP_COUNTS_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round3_na22_127_step_counts.png"
)
ROUND3_NA22_CENTROIDS_MID_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round3_na22_centroids_mid.png"
)
ROUND3_NA22_127_CENTROIDS_TOP_SAVE_FILE_PATH = os.path.join(
    ASSETS_PATH, "round3_na22_127_centroids_top.png"
)

def solid_angle_cone(theta):
    """
    Solid angle of a cone with apex diameter 2 * theta.
    """
    return 2 * np.pi * (1 - np.cos(theta))
#ENDDEF

tup_n_dn = lambda n: (n, np.sqrt(n))


# data constants - round 0
# obtained from fit
ROUND0_NA22_127_STEP_NET_COUNTS = np.array([
    (640, 40),
    (730, 40),
    (610, 40),
    (723, 40),
    (760, 40),
    (780, 60),
    (870, 60),
    (700, 40),
    (760, 50),
])
# from plot_spectrum.py
ROUND0_NA22_127_STEP_GROSS_COUNTS = np.array([
    tup_n_dn(843),
    tup_n_dn(919),
    tup_n_dn(850),
    tup_n_dn(874),
    tup_n_dn(912),
    tup_n_dn(886),
    tup_n_dn(925),
    tup_n_dn(896),
    tup_n_dn(859),
])

# data constants - round 1
# obtained from fit
ROUND1_NA22_127_STEP_NET_COUNTS = np.array([
    (3250, 70),
    (3140, 70),
    (3130, 70),
    (3240, 70),
    (3190, 70),
])
# from plot_spectrum.py
ROUND1_NA22_127_STEP_GROSS_COUNTS = np.array([
    tup_n_dn(3381),
    tup_n_dn(3278),
    tup_n_dn(3279),
    tup_n_dn(3417),
    tup_n_dn(3371),
])

# data constants - round 2
# obtained from fit
ROUND2_NA22_127_STEP_NET_COUNTS = np.array([
    (3050, 70),
    (2830, 70),
    (2930, 60),
    (3080, 70),
    (2940, 70),
])
# from plot_spectrum.py
ROUND2_NA22_127_STEP_GROSS_COUNTS = np.array([
    tup_n_dn(3197),
    tup_n_dn(3075),
    tup_n_dn(3112),
    tup_n_dn(3198),
    tup_n_dn(3117),
])

# data constants - round 3
# 6mm...0.5mm
ROUND3_NA22_127_MID_PEAKS = np.array([
    (716.8, 0.4),
    (717.0, 0.4),
    (717.4, 0.4),
    (717.0, 0.4),
    (716.8, 0.4),
    (716.8, 0.4),
    (715.7, 0.4),
])
ROUND3_NA22_511_MID_PEAKS = np.array([
    (278.2, 0.1),
    (278.1, 0.1),
    (278.4, 0.1),
    (278.5, 0.1),
    (278.4, 0.1),
    (278.5, 0.1),
    (278.5, 0.1),
])
# 0mm, 6mm...0.5mm
ROUND3_NA22_127_TOP_PEAKS = np.array([
    (736.8, 0.3),
    (730.8, 0.4),
    (727.0, 0.4),
    (720.9, 0.4),
    (720.6, 0.4),
    (718.9, 0.4),
    (718.0, 0.4),
    (718.3, 0.4),
])

# data constants - round 0 - apparatus (lengths in 1e-3 m)
# SA - solid angle
# SAR - solid angle ratio
# absorber diameter
DA = 50.81
DDA = 0.005
# detector diameter
DD = 44.55
DDD = 0.005
# height from source to detector
H0 = 327
DH0 = 2
THETA0 = np.arctan(DD / (2 * H0))
DTHETA0 = None
SOLID_ANGLE_0 = solid_angle_cone(THETA0)
DSOLID_ANGLE_0 = None
SOLID_ANGLE_RATIO_0 = SOLID_ANGLE_0 / (4 * np.pi)
# count rate for Na22 calibration round0_CalibrationNa22Gain3Time60s
# 60s live time, channels 730-830, 230 net counts for 1.27mev peak
GROSS_DETECTOR_COUNT_0 = 230 - (83 / 5)
DGROSS_DETECTOR_COUNT_0 = np.sqrt(GROSS_DETECTOR_COUNT_0)
# step heights
ROUND0_HEIGHTS = np.array([44, 77, 110, 143, 176, 209, 242, 275, 308])
ROUND0_DHEIGHTS = np.array([2,  2,   2,   2,   2,   2,   2,   2,   2])
ROUND0_THETAS = np.arctan(DA / (2 * ROUND0_HEIGHTS))
ROUND0_DTHETAS = None 
ROUND0_SOLID_ANGLES = solid_angle_cone(ROUND0_THETAS)
ROUND0_DSOLID_ANGLES = None
ROUND0_SOLID_ANGLE_RATIOS = ROUND0_SOLID_ANGLES / (4 * np.pi)
ROUND0_DSOLID_ANGLE_RATIOS = None
print(ROUND0_SOLID_ANGLE_RATIOS)
# 300s live time / 60s live time = 5
ROUND0_GROSS_ABSORBER_COUNTS = ROUND0_SOLID_ANGLES * 5 * GROSS_DETECTOR_COUNT_0 / SOLID_ANGLE_0
ROUND0_DGROSS_ABSORBER_COUNTS = None
ROUND0_NET_DETECTOR_COUNTS = ROUND0_NA22_127_STEP_NET_COUNTS[:, 0]
ROUND0_DNET_DETECTOR_COUNTS = None
ROUND0_RS = ROUND0_GROSS_ABSORBER_COUNTS / ROUND0_NET_DETECTOR_COUNTS

# data constants - round 1 apparatus (lengths in 1e-3 m)
ROUND2_HEIGHTS = ROUND1_HEIGHTS = np.array([0, 46, 90, 135, 154])
ROUND2_DHEIGHTS = ROUND1_DHEIGHTS = np.array([2, 2,  2,   2,   2])

# data constants - round 3 apparatus (lengths in 1e-3 m)
ROUND3_THICKNESSES = np.array([5.98, 4.52, 3.04, 1.96, 1.55, 1.00, 0.49, 0,])
ROUND3_DTHICKNESSES = np.array([1e-2] * len(ROUND3_THICKNESSES))
ROUND3_TOP_TIMES = np.array([
    (1351, 1953),
    (2283, 2584),
    (2951, 3252),
    (6396, 6697),
    (6908, 7209),
    (7607, 7908),
    (8141, 8442),
    (8574, 8875),
])
DTIME = 0.01 #s


# data constants - gaussian fits - round 0
ROUND0_NA22_127_STEP_INITIAL_PARAMETERS = [200., 750., 10., 1., 1.]
ROUND0_NA22_127_H1_CHANNEL_LO = 720
ROUND0_NA22_127_H1_CHANNEL_HI = 830
ROUND0_NA22_127_H2_CHANNEL_LO = 725
ROUND0_NA22_127_H2_CHANNEL_HI = 825
ROUND0_NA22_127_H3_CHANNEL_LO = 720
ROUND0_NA22_127_H3_CHANNEL_HI = 820
ROUND0_NA22_127_H4_CHANNEL_LO = 720
ROUND0_NA22_127_H4_CHANNEL_HI = 820
ROUND0_NA22_127_H5_CHANNEL_LO = 720
ROUND0_NA22_127_H5_CHANNEL_HI = 820
ROUND0_NA22_127_H6_CHANNEL_LO = 725
ROUND0_NA22_127_H6_CHANNEL_HI = 820
ROUND0_NA22_127_H7_CHANNEL_LO = 720
ROUND0_NA22_127_H7_CHANNEL_HI = 820
ROUND0_NA22_127_H8_CHANNEL_LO = 720
ROUND0_NA22_127_H8_CHANNEL_HI = 820
ROUND0_NA22_127_H9_CHANNEL_LO = 720
ROUND0_NA22_127_H9_CHANNEL_HI = 820
ROUND0_NA22_127_H1_PLOT_TEXT = (
    680,
    25,
    "$N = 640 \pm 40$ counts\n"
    "$\mu = 775 \pm 1$ channel\n"
    "$\sigma = 16 \pm 1$ channel\n"
    "$A = 8 \pm 4$ counts\n"
    "$B = (-8 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.28$\n",
)
ROUND0_NA22_127_H2_PLOT_TEXT = (
    680,
    20,
    "$N = 730 \pm 40$ counts\n"
    "$\mu = 774 \pm 1$ channel\n"
    "$\sigma = 15 \pm 1$ channel\n"
    "$A = 17 \pm 5$ counts\n"
    "$B = (-2.0 \pm 0.6) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.95$\n",
)
ROUND0_NA22_127_H3_PLOT_TEXT = (
    680,
    20,
    "$N = 610 \pm 40$ counts\n"
    "$\mu = 770 \pm 1$ channel\n"
    "$\sigma = 15 \pm 1$ channel\n"
    "$A = 12 \pm 5$ counts\n"
    "$B = (-1.4 \pm 0.6) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.30$\n",
)
ROUND0_NA22_127_H4_PLOT_TEXT = (
    680,
    20,
    "$N = 723 \pm 40$ counts\n"
    "$\mu = 772 \pm 1$ channel\n"
    "$\sigma = 16 \pm 1$ channel\n"
    "$A = 2 \pm 4$ counts\n"
    "$B = (-2 \pm 5) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.93$\n",
)
ROUND0_NA22_127_H5_PLOT_TEXT = (
    680,
    20,
    "$N = 760 \pm 40$ counts\n"
    "$\mu = 772 \pm 1$ channel\n"
    "$\sigma = 15 \pm 1$ channel\n"
    "$A = 7 \pm 4$ counts\n"
    "$B = (-8 \pm 6) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.86$\n",
)
ROUND0_NA22_127_H6_PLOT_TEXT = (
    680,
    20,
    "$N = 780 \pm 60$ counts\n"
    "$\mu = 771 \pm 1$ channel\n"
    "$\sigma = 18 \pm 1$ channel\n"
    "$A = 2 \pm 6$ counts\n"
    "$B = (-2 \pm 8) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.12$\n",
)
ROUND0_NA22_127_H7_PLOT_TEXT = (
    680,
    20,
    "$N = 870 \pm 60$ counts\n"
    "$\mu = 771 \pm 1$ channel\n"
    "$\sigma = 18 \pm 1$ channel\n"
    "$A = 3 \pm 5$ counts\n"
    "$B = (-3 \pm 6) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.81$\n",
)
ROUND0_NA22_127_H8_PLOT_TEXT = (
    680,
    20,
    "$N = 700 \pm 40$ counts\n"
    "$\mu = 771 \pm 1$ channel\n"
    "$\sigma = 16 \pm 1$ channel\n"
    "$A = 9 \pm 5$ counts\n"
    "$B = (-9 \pm 6) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.94$\n",
)
ROUND0_NA22_127_H9_PLOT_TEXT = (
    680,
    20,
    "$N = 760 \pm 50$ counts\n"
    "$\mu = 771 \pm 1$ channel\n"
    "$\sigma = 17 \pm 1$ channel\n"
    "$A = 7 \pm 4$ counts\n"
    "$B = (-9 \pm 6) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.86$\n",
)
ROUND0_NA22_127_STEP_DATA = [
    (
        ROUND0_NA22_127_H1_CHANNEL_LO,
        ROUND0_NA22_127_H1_CHANNEL_HI,
        ROUND0_NA22_127_H1_PLOT_TEXT,
    ),
    (
        ROUND0_NA22_127_H2_CHANNEL_LO,
        ROUND0_NA22_127_H2_CHANNEL_HI,
        ROUND0_NA22_127_H2_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H3_CHANNEL_LO,
        ROUND0_NA22_127_H3_CHANNEL_HI,
        ROUND0_NA22_127_H3_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H4_CHANNEL_LO,
        ROUND0_NA22_127_H4_CHANNEL_HI,
        ROUND0_NA22_127_H4_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H5_CHANNEL_LO,
        ROUND0_NA22_127_H5_CHANNEL_HI,
        ROUND0_NA22_127_H5_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H6_CHANNEL_LO,
        ROUND0_NA22_127_H6_CHANNEL_HI,
        ROUND0_NA22_127_H6_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H7_CHANNEL_LO,
        ROUND0_NA22_127_H7_CHANNEL_HI,
        ROUND0_NA22_127_H7_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H8_CHANNEL_LO,
        ROUND0_NA22_127_H8_CHANNEL_HI,
        ROUND0_NA22_127_H8_PLOT_TEXT,
    ),
        (
        ROUND0_NA22_127_H9_CHANNEL_LO,
        ROUND0_NA22_127_H9_CHANNEL_HI,
        ROUND0_NA22_127_H9_PLOT_TEXT,
    ),
]

# data - gaussian fits - round 1
ROUND1_NA22_127_STEP_INITIAL_PARAMETERS = [600., 750., 10., 10., 1.]
ROUND1_NA22_127_H1_CHANNEL_LO = 650
ROUND1_NA22_127_H1_CHANNEL_HI = 785
ROUND1_NA22_127_H2_CHANNEL_LO = 650
ROUND1_NA22_127_H2_CHANNEL_HI = 785
ROUND1_NA22_127_H3_CHANNEL_LO = 650
ROUND1_NA22_127_H3_CHANNEL_HI = 785
ROUND1_NA22_127_H4_CHANNEL_LO = 650
ROUND1_NA22_127_H4_CHANNEL_HI = 785
ROUND1_NA22_127_H5_CHANNEL_LO = 650
ROUND1_NA22_127_H5_CHANNEL_HI = 785
ROUND1_NA22_127_H1_PLOT_TEXT = (
    605,
    60,
    "$N = (3.25 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 718.9 \pm 0.4$ channel\n"
    "$\sigma = 19.2 \pm 0.3$ channel\n"
    "$A = 14 \pm 3$ counts\n"
    "$B = (-1.9 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.81$\n",
)
ROUND1_NA22_127_H2_PLOT_TEXT = (
    605,
    60,
    "$N = (3.14 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 717.0 \pm 0.4$ channel\n"
    "$\sigma = 19.5 \pm 0.4$ channel\n"
    "$A = 9 \pm 3$ counts\n"
    "$B = (-1.2 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.76$\n",
)
ROUND1_NA22_127_H3_PLOT_TEXT = (
    605,
    60,
    "$N = (3.13 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 716.8 \pm 0.4$ channel\n"
    "$\sigma = 19.3 \pm 0.4$ channel\n"
    "$A = 7 \pm 3$ counts\n"
    "$B = (-9 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.01$\n",
)
ROUND1_NA22_127_H4_PLOT_TEXT = (
    605,
    60,
    "$N = (3.24 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 717.4 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 9 \pm 3$ counts\n"
    "$B = (-1.1 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.99$\n",
)
ROUND1_NA22_127_H5_PLOT_TEXT = (
    605,
    60,
    "$N = (3.19 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 718.8 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 7 \pm 3$ counts\n"
    "$B = (-9 \pm 6) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.98$\n",
)
ROUND1_NA22_127_STEP_DATA = [
    (
        ROUND1_NA22_127_H1_CHANNEL_LO,
        ROUND1_NA22_127_H1_CHANNEL_HI,
        ROUND1_NA22_127_H1_PLOT_TEXT,
    ),
    (
        ROUND1_NA22_127_H2_CHANNEL_LO,
        ROUND1_NA22_127_H2_CHANNEL_HI,
        ROUND1_NA22_127_H2_PLOT_TEXT,
    ),
        (
        ROUND1_NA22_127_H3_CHANNEL_LO,
        ROUND1_NA22_127_H3_CHANNEL_HI,
        ROUND1_NA22_127_H3_PLOT_TEXT,
    ),
        (
        ROUND1_NA22_127_H4_CHANNEL_LO,
        ROUND1_NA22_127_H4_CHANNEL_HI,
        ROUND1_NA22_127_H4_PLOT_TEXT,
    ),
        (
        ROUND1_NA22_127_H5_CHANNEL_LO,
        ROUND1_NA22_127_H5_CHANNEL_HI,
        ROUND1_NA22_127_H5_PLOT_TEXT,
    ),
 ]

# data - gaussian fits - round 2
ROUND2_NA22_127_STEP_INITIAL_PARAMETERS = [600., 750., 10., 10., 1.]
ROUND2_NA22_127_H1_CHANNEL_LO = 650
ROUND2_NA22_127_H1_CHANNEL_HI = 785
ROUND2_NA22_127_H2_CHANNEL_LO = 650
ROUND2_NA22_127_H2_CHANNEL_HI = 785
ROUND2_NA22_127_H3_CHANNEL_LO = 650
ROUND2_NA22_127_H3_CHANNEL_HI = 785
ROUND2_NA22_127_H4_CHANNEL_LO = 650
ROUND2_NA22_127_H4_CHANNEL_HI = 785
ROUND2_NA22_127_H5_CHANNEL_LO = 665
ROUND2_NA22_127_H5_CHANNEL_HI = 800
ROUND2_NA22_127_H1_PLOT_TEXT = (
    605,
    60,
    "$N = (3.05 \pm 0.06) 10^{3}$ counts\n"
    "$\mu = 717.9 \pm 0.4$ channel\n"
    "$\sigma = 18.9 \pm 0.4$ channel\n"
    "$A = 11 \pm 3$ counts\n"
    "$B = (-1.4 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.93$\n",
)
ROUND2_NA22_127_H2_PLOT_TEXT = (
    605,
    60,
    "$N = (2.83 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 716.6 \pm 0.4$ channel\n"
    "$\sigma = 18.9 \pm 0.4$ channel\n"
    "$A = 15 \pm 3$ counts\n"
    "$B = (-1.9 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.03$\n",
)
ROUND2_NA22_127_H3_PLOT_TEXT = (
    605,
    60,
    "$N = (2.93 \pm 0.06) 10^{3}$ counts\n"
    "$\mu = 716.8 \pm 0.4$ channel\n"
    "$\sigma = 18.8 \pm 0.4$ channel\n"
    "$A = 15 \pm 3$ counts\n"
    "$B = (-1.9 \pm 0.04) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.79$\n",
)
ROUND2_NA22_127_H4_PLOT_TEXT = (
    605,
    60,
    "$N = (3.08 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 717.8 \pm 0.4$ channel\n"
    "$\sigma = 19.6 \pm 0.4$ channel\n"
    "$A = 6 \pm 3$ counts\n"
    "$B = (-7 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.72$\n",
)
ROUND2_NA22_127_H5_PLOT_TEXT = (
    620,
    60,
    "$N = (2.94 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 730.7 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 7 \pm 3$ counts\n"
    "$B = (-9 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.19$\n",
)
ROUND2_NA22_127_STEP_DATA = [
    (
        ROUND2_NA22_127_H1_CHANNEL_LO,
        ROUND2_NA22_127_H1_CHANNEL_HI,
        ROUND2_NA22_127_H1_PLOT_TEXT,
    ),
    (
        ROUND2_NA22_127_H2_CHANNEL_LO,
        ROUND2_NA22_127_H2_CHANNEL_HI,
        ROUND2_NA22_127_H2_PLOT_TEXT,
    ),
        (
        ROUND2_NA22_127_H3_CHANNEL_LO,
        ROUND2_NA22_127_H3_CHANNEL_HI,
        ROUND2_NA22_127_H3_PLOT_TEXT,
    ),
        (
        ROUND2_NA22_127_H4_CHANNEL_LO,
        ROUND2_NA22_127_H4_CHANNEL_HI,
        ROUND2_NA22_127_H4_PLOT_TEXT,
    ),
        (
        ROUND2_NA22_127_H5_CHANNEL_LO,
        ROUND2_NA22_127_H5_CHANNEL_HI,
        ROUND2_NA22_127_H5_PLOT_TEXT,
    ),
 ]

# data constants - gaussian fits - round 3
ROUND3_NA22_127_STEP_INITIAL_PARAMETERS = [200., 750., 10., 1., 1.]
# CALIBRATION
ROUND3_NA22_127_0MM_CHANNEL_LO = 670
ROUND3_NA22_127_0MM_CHANNEL_HI = 800
# MID
ROUND3_NA22_127_6MM_MID_CHANNEL_LO = 650
ROUND3_NA22_127_6MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_45MM_MID_CHANNEL_LO = 650
ROUND3_NA22_127_45MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_3MM_MID_CHANNEL_LO = 650
ROUND3_NA22_127_3MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_2MM_MID_CHANNEL_LO = 655
ROUND3_NA22_127_2MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_15MM_MID_CHANNEL_LO = 655
ROUND3_NA22_127_15MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_1MM_MID_CHANNEL_LO = 655
ROUND3_NA22_127_1MM_MID_CHANNEL_HI = 785
ROUND3_NA22_127_05MM_MID_CHANNEL_LO = 655
ROUND3_NA22_127_05MM_MID_CHANNEL_HI = 785
# TOP
ROUND3_NA22_127_6MM_TOP_CHANNEL_LO = 665
ROUND3_NA22_127_6MM_TOP_CHANNEL_HI = 795
ROUND3_NA22_127_45MM_TOP_CHANNEL_LO = 665
ROUND3_NA22_127_45MM_TOP_CHANNEL_HI = 795
ROUND3_NA22_127_3MM_TOP_CHANNEL_LO = 665
ROUND3_NA22_127_3MM_TOP_CHANNEL_HI = 785
ROUND3_NA22_127_2MM_TOP_CHANNEL_LO = 660
ROUND3_NA22_127_2MM_TOP_CHANNEL_HI = 790
ROUND3_NA22_127_15MM_TOP_CHANNEL_LO = 655
ROUND3_NA22_127_15MM_TOP_CHANNEL_HI = 785
ROUND3_NA22_127_1MM_TOP_CHANNEL_LO = 655
ROUND3_NA22_127_1MM_TOP_CHANNEL_HI = 785
ROUND3_NA22_127_05MM_TOP_CHANNEL_LO = 655
ROUND3_NA22_127_05MM_TOP_CHANNEL_HI = 785
ROUND3_NA22_127_0MM_PLOT_TEXT = (
    630,
    100,
    "$N = (6.5 \pm 0.1) 10^{3}$ counts\n"
    "$\mu = 736.8 \pm 0.3$ channel\n"
    "$\sigma = 19.4 \pm 0.3$ channel\n"
    "$A = 20 \pm 4$ counts\n"
    "$B = (-2.5 \pm 0.6) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.08$\n",
)
ROUND3_NA22_127_6MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (2.93 \pm 0.06) 10^{3}$ counts\n"
    "$\mu = 716.8 \pm 0.4$ channel\n"
    "$\sigma = 18.8 \pm 0.4$ channel\n"
    "$A = 15 \pm 3$ counts\n"
    "$B = (-1.9 \pm 0.3) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.79$\n",
)
ROUND3_NA22_127_45MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (2.93 \pm 0.06) 10^{3}$ counts\n"
    "$\mu = 717.0 \pm 0.4$ channel\n"
    "$\sigma = 19.0 \pm 0.4$ channel\n"
    "$A = 11 \pm 3$ counts\n"
    "$B = (-1.4 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.89$\n",
)
ROUND3_NA22_127_3MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (3.01 \pm 0.06) 10^{3}$ counts\n"
    "$\mu = 717.4 \pm 0.4$ channel\n"
    "$\sigma = 18.3 \pm 0.4$ channel\n"
    "$A = 8 \pm 3$ counts\n"
    "$B = (-1.0 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.98$\n",
)
ROUND3_NA22_127_2MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (3.02 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 717.0 \pm 0.4$ channel\n"
    "$\sigma = 18.6 \pm 0.4$ channel\n"
    "$A = 4 \pm 3$ counts\n"
    "$B = (-4 \pm 5) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.22$\n",
)
ROUND3_NA22_127_15MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (3.13 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 716.8 \pm 0.4$ channel\n"
    "$\sigma = 19.3 \pm 0.4$ channel\n"
    "$A = 7 \pm 3$ counts\n"
    "$B = (-9 \pm 5) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.02$\n",
)
ROUND3_NA22_127_1MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (3.12 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 716.8 \pm 0.4$ channel\n"
    "$\sigma = 18.9 \pm 0.4$ channel\n"
    "$A = 6 \pm 3$ counts\n"
    "$B = (-8 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.96$\n",
)
ROUND3_NA22_127_05MM_MID_PLOT_TEXT = (
    610,
    50,
    "$N = (3.27 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 715.7 \pm 0.4$ channel\n"
    "$\sigma = 18.9 \pm 0.4$ channel\n"
    "$A = 5 \pm 3$ counts\n"
    "$B = (-6 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.07$\n",
)
ROUND3_NA22_127_6MM_TOP_PLOT_TEXT = (
    630,
    65,
    "$N = (2.95 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 730.8 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 8 \pm 3$ counts\n"
    "$B = (-1.0 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.22$\n",
)
ROUND3_NA22_127_45MM_TOP_PLOT_TEXT = (
    630,
    60,
    "$N = (3.10 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 727.0 \pm 0.4$ channel\n"
    "$\sigma = 19.7 \pm 0.4$ channel\n"
    "$A = 7 \pm 3$ counts\n"
    "$B = (-9 \pm 4) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.75$\n",
)
ROUND3_NA22_127_3MM_TOP_PLOT_TEXT = (
    630,
    60,
    "$N = (2.97 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 720.9 \pm 0.4$ channel\n"
    "$\sigma = 18.5 \pm 0.4$ channel\n"
    "$A = 13 \pm 5$ counts\n"
    "$B = (-1.6 \pm 0.7) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.98$\n",
)
ROUND3_NA22_127_2MM_TOP_PLOT_TEXT = (
    630,
    55,
    "$N = (3.14 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 720.6 \pm 0.4$ channel\n"
    "$\sigma = 19.0 \pm 0.4$ channel\n"
    "$A = 6 \pm 4$ counts\n"
    "$B = (-8 \pm 5) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.15$\n",
)
ROUND3_NA22_127_15MM_TOP_PLOT_TEXT = (
    610,
    60,
    "$N = (3.19 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 718.9 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 8 \pm 3$ counts\n"
    "$B = (-9 \pm 5) 10^{-3}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.97$\n",
)
ROUND3_NA22_127_1MM_TOP_PLOT_TEXT = (
    610,
    60,
    "$N = (3.02 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 718.0 \pm 0.4$ channel\n"
    "$\sigma = 18.8 \pm 0.4$ channel\n"
    "$A = 10 \pm 4$ counts\n"
    "$B = (-1.2 \pm 0.5) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 1.15$\n",
)
ROUND3_NA22_127_05MM_TOP_PLOT_TEXT = (
    610,
    60,
    "$N = (3.13 \pm 0.07) 10^{3}$ counts\n"
    "$\mu = 718.3 \pm 0.4$ channel\n"
    "$\sigma = 19.4 \pm 0.4$ channel\n"
    "$A = 11 \pm 4$ counts\n"
    "$B = (-1.4 \pm 0.4) 10^{-2}$ counts/channel\n"
    "$\widetilde{\chi^{2}} = 0.80$\n",
)
ROUND3_NA22_127_STEP_DATA = [
    (
        ROUND3_NA22_127_0MM_CHANNEL_LO,
        ROUND3_NA22_127_0MM_CHANNEL_HI,
        ROUND3_NA22_127_0MM_PLOT_TEXT,
    ),
    (
        ROUND3_NA22_127_6MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_6MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_6MM_MID_PLOT_TEXT,
    ),
    (
        ROUND3_NA22_127_45MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_45MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_45MM_MID_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_3MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_3MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_3MM_MID_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_2MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_2MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_2MM_MID_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_15MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_15MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_15MM_MID_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_1MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_1MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_1MM_MID_PLOT_TEXT,
    ),
    (
        ROUND3_NA22_127_05MM_MID_CHANNEL_LO,
        ROUND3_NA22_127_05MM_MID_CHANNEL_HI,
        ROUND3_NA22_127_05MM_MID_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_6MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_6MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_6MM_TOP_PLOT_TEXT,
    ),
    (
        ROUND3_NA22_127_45MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_45MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_45MM_TOP_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_3MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_3MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_3MM_TOP_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_2MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_2MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_2MM_TOP_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_15MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_15MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_15MM_TOP_PLOT_TEXT,
    ),
        (
        ROUND3_NA22_127_1MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_1MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_1MM_TOP_PLOT_TEXT,
    ),
    (
        ROUND3_NA22_127_05MM_TOP_CHANNEL_LO,
        ROUND3_NA22_127_05MM_TOP_CHANNEL_HI,
        ROUND3_NA22_127_05MM_TOP_PLOT_TEXT,
    ),
]


# processing constants
TAB_DELIMITER = "\t"
DPI = int(1e3)
ROUND0_SKIPROWS = 23
ROUND3_SKIPROWS = ROUND2_SKIPROWS = ROUND1_SKIPROWS = 24
COLS = (0, 2)
COUNT_DTYPE = int
FIT_PARAMETER_COUNT = 5
PLOT_FIT_SAMPLES = int(1e3)
PLOT_CHANNEL_BUF = 40
PLOT_TEXT_FONT_SIZE = 8
MARKER_SIZE = 5


def linear_fitfunc(p, x):
    """
    Fit to a linear function with constant background:
    f(x) = A * x + B
    A = p[0]
    B = p[1]
    """
    return (
        p[0] * x + p[1]
    )
#ENDDEF


def linear_residual(p, x, y, yerr):
    """
    Residual for linear_fitfunc that incorporates y error.
    """
    yp = linear_fitfunc(p, x)
    return (yp - y) / yerr
#ENDDEF


def gaussian_fitfunc(p, x):
    """
    Fit to a Gaussian with linear background:
    \frac{N_{0}}{\sigma \sqrt{2 \pi}} \exp{\frac{-(x - \mu)^{2}}{2 {\sigma}^{2}}} + A + Bx
    """
    return (
        (p[0] / (p[2] * np.sqrt(2 * np.pi))) * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))
            + p[3] + p[4] * x
    )
#ENDDEF


def gaussian_residual(p, x, y, yerr):
    """
    Use a simple residual that incorporates y error.
    """
    yp = gaussian_fitfunc(p, x)
    return (yp - y) / yerr
#ENDDEF


def gaussian_fit_and_plot(data_file_path,
                          channel_lo=0, channel_hi=1000,
                          initial_parameters=None,
                          plot_title=None,
                          plot_text=None,
                          save_file_path=None,
                          skiprows=0):
    """
    Perform a gaussian fit to the specified data and the specified region.
    
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
        skiprows=skiprows,
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
        gaussian_residual, initial_parameters, args=(channels_slice, counts_slice, dcounts_slice),
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
        counts_fit = gaussian_fitfunc(pf, channels_fit)
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
#ENDDEF


def plot_step_counts(net_counts, dnet_counts,
                     gross_counts, dgross_counts,
                     heights, dheights,
                     save_file_path,
                     title=None):
    """
    Plot the net count obtained from the Na22 1.27 MeV peak fit for each
    step measurement.
    
    Arguments: None
    Returns: None
    """
    plt.figure()
    plt.errorbar(heights, net_counts, xerr=dheights,
                 yerr=dnet_counts, linestyle="None",
                 fmt=".", ms=MARKER_SIZE, color="blue", label="Net")
    plt.errorbar(heights, gross_counts, xerr=dheights,
                 yerr=dgross_counts, linestyle="None",
                 fmt=".", ms=MARKER_SIZE, color="red", label="Gross")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Na22 - 1.27 MeV - Counts")
    plt.ylabel("Counts")
    plt.xlabel("Distance from Source ($10^{-3}$ m)")
    plt.legend()
    plt.savefig(save_file_path,
                dpi=DPI)
    return
#ENDDEF


def plot_round3_mid_centroids():
    thicknesses = ROUND3_THICKNESSES[:-1]
    dthicknesses = ROUND3_DTHICKNESSES[:-1]
    centroids_127 = ROUND3_NA22_127_MID_PEAKS[:, 0]
    dcentroids_127 = ROUND3_NA22_127_MID_PEAKS[:, 1]
    centroids_511 = ROUND3_NA22_511_MID_PEAKS[:, 0]
    dcentroids_511 = ROUND3_NA22_511_MID_PEAKS[:, 1]
    ylabel = "Centroid (Channel)"
    xlabel = "Absorber Thickness ($10^{-3}$ m)"
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    plt.errorbar(thicknesses, centroids_127, xerr=dthicknesses,
                 yerr=dcentroids_127, linestyle="None",
                 fmt=".", ms=MARKER_SIZE, color="blue",
                 label="1.27 MeV")
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax2 = plt.subplot(2, 1, 2)
    plt.errorbar(thicknesses, centroids_511, xerr=dthicknesses,
                 yerr=dcentroids_511, linestyle="None",
                 fmt=".", ms=MARKER_SIZE, color="red",
                 label="511 keV")
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    ax2.legend()
    plt.suptitle("Na22 - MID - Centroids")
    plt.savefig(ROUND3_NA22_CENTROIDS_MID_SAVE_FILE_PATH,
                dpi=DPI)
    return
#ENDDEF


def plot_round3_top_centroids():
    centroids = ROUND3_NA22_127_TOP_PEAKS[:, 0]
    dcentroids = ROUND3_NA22_127_TOP_PEAKS[:, 1]
    times = ROUND3_TOP_TIMES[:, 0]
    save_file_path = ROUND3_NA22_127_CENTROIDS_TOP_SAVE_FILE_PATH
    plt.figure()
    plt.errorbar(times, centroids, yerr=dcentroids, color="blue",
                 fmt=".", ms=MARKER_SIZE, linestyle="None")
    plt.title("Na22 - 1.27 MeV - TOP - Centroids")
    plt.ylabel("Centroid (Channel)")
    plt.xlabel("Final Time (s)")
    plt.savefig(save_file_path,
                dpi=DPI)
#ENDDEF


def main():
    """
    Do all the things.
    """
    # Round 0 - Na22 - 1.27 MeV - Height Steps
    if False:
        # all
        round0_na22_127_step_indices = np.arange(ROUND0_STEP_COUNT)
        # selected
        # round0_na22_127_step_indices = np.array([
        #     2,
        # ])
        for i in round0_na22_127_step_indices:
            (channel_lo, channel_hi, plot_text) = ROUND0_NA22_127_STEP_DATA[i]
            gaussian_fit_and_plot(
                ROUND0_NA22_STEP_DATA_FILE_PATHS[i],
                channel_hi=channel_hi,
                channel_lo=channel_lo,
                initial_parameters=ROUND0_NA22_127_STEP_INITIAL_PARAMETERS,
                plot_text=plot_text,
                plot_title=ROUND0_NA22_127_STEP_PLOT_TITLES[i],
                # save_file_path=ROUND0_NA22_127_STEP_SAVE_FILE_PATHS[i],
                skiprows=ROUND0_SKIPROWS,
            )
        #ENDFOR
    #ENDIF
    if False:
        plot_step_counts(ROUND0_NA22_127_STEP_NET_COUNTS[:, 0],
                         ROUND0_NA22_127_STEP_NET_COUNTS[:, 1],
                         ROUND0_NA22_127_STEP_GROSS_COUNTS[:, 0],
                         ROUND0_NA22_127_STEP_GROSS_COUNTS[:, 1],
                         ROUND0_HEIGHTS,
                         ROUND0_DHEIGHTS,
                         ROUND0_NA22_127_STEP_COUNTS_SAVE_FILE_PATH,
                         title="Na22 - 1.27 MeV - Experiment 1 - Counts"
        )
    #ENDIF

    # Round 1
    if False:
        round1_na22_127_step_indices = np.arange(ROUND1_STEP_COUNT)
        # round1_na22_127_step_indices = np.array([
        #     4
        # ])
        for i in round1_na22_127_step_indices:
            (channel_lo, channel_hi, plot_text) = ROUND1_NA22_127_STEP_DATA[i]
            gaussian_fit_and_plot(
                ROUND1_NA22_STEP_DATA_FILE_PATHS[i],
                channel_hi=channel_hi,
                channel_lo=channel_lo,
                initial_parameters=ROUND1_NA22_127_STEP_INITIAL_PARAMETERS,
                plot_text=plot_text,
                plot_title=ROUND1_NA22_127_STEP_PLOT_TITLES[i],
                save_file_path=ROUND1_NA22_127_STEP_SAVE_FILE_PATHS[i],
                skiprows=ROUND1_SKIPROWS,
            )
        #ENDFOR
    #ENDIF
    if False:
        plot_step_counts(ROUND1_NA22_127_STEP_NET_COUNTS[:, 0],
                         ROUND1_NA22_127_STEP_NET_COUNTS[:, 1],
                         ROUND1_NA22_127_STEP_GROSS_COUNTS[:, 0],
                         ROUND1_NA22_127_STEP_GROSS_COUNTS[:, 1],
                         ROUND1_HEIGHTS,
                         ROUND1_DHEIGHTS,
                         ROUND1_NA22_127_STEP_COUNTS_SAVE_FILE_PATH,
                         title="Na22 - 1.27 MeV - Experiment 2 - Counts",
        )
    #ENDIF

    # Round 2
    if False:
        # round2_na22_127_step_indices = np.arange(ROUND2_STEP_COUNT)
        round2_na22_127_step_indices = np.array([
            0,
        ])
        for i in round2_na22_127_step_indices:
            (channel_lo, channel_hi, plot_text) = ROUND2_NA22_127_STEP_DATA[i]
            gaussian_fit_and_plot(
                ROUND2_NA22_STEP_DATA_FILE_PATHS[i],
                channel_hi=channel_hi,
                channel_lo=channel_lo,
                initial_parameters=ROUND2_NA22_127_STEP_INITIAL_PARAMETERS,
                plot_text=plot_text,
                plot_title=ROUND2_NA22_127_STEP_PLOT_TITLES[i],
                save_file_path=ROUND2_NA22_127_STEP_SAVE_FILE_PATHS[i],
                skiprows=ROUND2_SKIPROWS,
            )
        #ENDFOR
    #ENDIF
    if False:
        plot_step_counts(ROUND2_NA22_127_STEP_NET_COUNTS[:, 0],
                         ROUND2_NA22_127_STEP_NET_COUNTS[:, 1],
                         ROUND2_NA22_127_STEP_GROSS_COUNTS[:, 0],
                         ROUND2_NA22_127_STEP_GROSS_COUNTS[:, 1],
                         ROUND2_HEIGHTS,
                         ROUND2_DHEIGHTS,
                         ROUND2_NA22_127_STEP_COUNTS_SAVE_FILE_PATH,
                         title="Na22 - 1.27 MeV - Experiment 2 - Counts",
        )
    #ENDIF

    # Round 3
    if False:
        # round3_na22_127_step_indices = np.arange(ROUND3_STEP_COUNT)
        round3_na22_127_step_indices = np.array([
            0,
        ])
        for i in round3_na22_127_step_indices:
            (channel_lo, channel_hi, plot_text) = ROUND3_NA22_127_STEP_DATA[i]
            gaussian_fit_and_plot(
                ROUND3_NA22_STEP_DATA_FILE_PATHS[i],
                channel_hi=channel_hi,
                channel_lo=channel_lo,
                initial_parameters=ROUND3_NA22_127_STEP_INITIAL_PARAMETERS,
                plot_text=plot_text,
                plot_title=ROUND3_NA22_127_STEP_PLOT_TITLES[i],
                save_file_path=ROUND3_NA22_127_STEP_SAVE_FILE_PATHS[i],
                skiprows=ROUND3_SKIPROWS,
            )
        #ENDFOR
    #ENDIF
    if False:
        plot_step_counts(ROUND3_NA22_127_STEP_NET_COUNTS[:, 0],
                         ROUND3_NA22_127_STEP_NET_COUNTS[:, 1],
                         ROUND3_NA22_127_STEP_GROSS_COUNTS[:, 0],
                         ROUND3_NA22_127_STEP_GROSS_COUNTS[:, 1],
                         ROUND3_HEIGHTS,
                         ROUND3_DHEIGHTS,
                         ROUND3_NA22_127_STEP_COUNTS_SAVE_FILE_PATH,
        )
    #ENDIF
    if False:
        plot_round3_mid_centroids()
    #ENDIF
    if False:
        plot_round3_top_centroids()
#ENDDEF


if __name__ == "__main__":
    main()
