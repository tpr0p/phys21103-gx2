"""
compton.py - This module does some analysis on compton scattering.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

WDIR = os.environ.get("PHYS_GX2_PATH", ".")
DATA_PATH = os.path.join(WDIR, "data")
ASSETS_PATH = os.path.join(WDIR, "assets")
NA22_127_KLEIN_NISHINA_SPECTRUM_SAVE_FILE_PATH = os.path.join(ASSETS_PATH, "na22_127_klein-nishina_spectrum.png")

# Energies in KeV. Masses in KeV / c^2.
electron_mass = 511
na127_peak_channels = np.array([775, 774, 770, 772, 772, 771, 771, 771, 771])
na127_peak_channel = np.mean(na127_peak_channels)
NA22_127_ENERGY = 1270
na511_peak_channels = np.array([318, 317.5, 317, 317.5, 317, 317.5, 316, 317, 316.5])
na511_peak_channel = np.mean(na511_peak_channels)
NA22_511_ENERGY = 511
na127_peak_channel_sigma = 15
NA22_127_MAX_RESOLVABLE_SCATTER_ANGLE = 0.2617993877991494
NA22_511_ENERGY = 511
CS137_32_ENERGY = 32
CS137_662_ENERGY = 662
BA133_356_ENERGY = 356
BA133_81_ENERGY = 81



# constants
# 1/2 * alpha^2 * r_c^2
KLEIN_NISHINA_PREFACTOR = 3.97039 # 10^-30 m

# processing constants
DPI = int(1e3)
SAMPLE_COUNT = int(5e2)
RAD_2_DEG = 180 / np.pi

def inverse_scatter_energy_ratio(energy, scatter_angle):
    """
    Returns what you think it does.
    """
    return (1 + (energy / electron_mass) * (1 - np.cos(scatter_angle)))


def scatter_energy(energy, scatter_angle):
    """
    Returns the scattered energy of a photon at incident energy `energy`
    that is scattered at an angle `scatter_angle` due to the compton effect.
    """
    return energy / inverse_scatter_energy_ratio(energy, scatter_angle)


def scatter_angle(energy, scatter_energy):
    """
    Returns the scattered angle of a photon at incident energy `energy`
    that is scattered at an energy `scatter_energy` due to the compton effect.
    """
    return np.arccos(1 - electron_mass * (1/scatter_energy - 1/energy))


def klein_nishina(energy, scatter_angle):
    """
    Calculate the differential cross-section for a photon at an incident `energy`
    and scattered at an angle `scatter_angle` due to the compton effect.
    """
    pinv = inverse_scatter_energy_ratio(energy, scatter_angle)
    pinv2 = pinv * pinv
    pinv3 = pinv2 * pinv
    return KLEIN_NISHINA_PREFACTOR * ((1 + pinv2 - pinv * np.sin(scatter_angle) ** 2) / pinv3)


def plot_klein_nishina():
    """
    Plot the klein nishina distribution for our sources.
    """
    scatter_angles = np.linspace(0, np.pi, SAMPLE_COUNT)
    scatter_angles_deg = scatter_angles * RAD_2_DEG
    na22_127_dc = klein_nishina(NA22_127_ENERGY, scatter_angles)
    na22_511_dc = klein_nishina(NA22_511_ENERGY, scatter_angles)
    cs137_662_dc = klein_nishina(CS137_662_ENERGY, scatter_angles)
    cs137_32_dc = klein_nishina(CS137_32_ENERGY, scatter_angles)
    ba133_356_dc = klein_nishina(BA133_356_ENERGY, scatter_angles)
    ba133_81_dc = klein_nishina(BA133_81_ENERGY, scatter_angles)
    plt.figure()
    plt.plot(scatter_angles_deg, na22_127_dc, color="blue", label="Na22 - 1.27 MeV")
    plt.plot(scatter_angles_deg, cs137_662_dc, color="purple", label="Cs137 - 662 KeV")
    plt.plot(scatter_angles_deg, na22_511_dc, color="red", label="Na22 - 511 KeV")
    plt.plot(scatter_angles_deg, ba133_356_dc, color="orange", label="Ba133 - 356 KeV")
    plt.plot(scatter_angles_deg, ba133_81_dc, color="brown", label="Ba133 - 81 KeV")
    plt.plot(scatter_angles_deg, cs137_32_dc, color="green", label="Cs137 - 32 KeV")
    plt.axvline(NA22_127_MAX_RESOLVABLE_SCATTER_ANGLE * RAD_2_DEG,
               label="Na22 1.27 MeV Max", color="blue")
    plt.ylabel("Differential Cross Section ($10^{-30}$ m)")
    plt.xlabel("Scatter Angle (Deg)")
    plt.title("Klein Nishina Distribution")
    plt.legend()
    plt.savefig(NA22_127_KLEIN_NISHINA_SPECTRUM_SAVE_FILE_PATH, dpi=DPI)


def main():
    plot_klein_nishina()


if __name__ == "__main__":
    main()


def fitfunc(p, x):
    """
    Fit to y(x) = Ax + B
    """
    return p[0] * x + p[1]


def residual(p, x, y):
    yp = fitfunc(p, x)
    return y - yp


def compute_max_scatter_angle():
    """
    Compute the maximum angle at which an Na22 1.27 MeV photon may be compton scattered
    and still end up in the full energy peak.
    """
    # Fit energies to channels.
    initial_parameters = [1., 1.]
    channels = np.array([na127_peak_channel, na511_peak_channel])
    energies = np.array([NA22_127_ENERGY, NA22_511_ENERGY])
    pf, cov, info, mesg, success = leastsq(
        residual, initial_parameters, args=(channels, energies),
        full_output=1,
    )
    chisq = sum(info["fvec"] ** 2)
    # dof = len(channels) - len(pf)
    # rchisq = chisq / dof
    pferr = [np.sqrt(cov[i, i]) for i in range(len(pf))]
    # print("p:\n{}\ndp:\n{}".format(pf, pferr))
    A = pf[0]

    na127_peak_channel_lo = na127_peak_channel - A * na127_peak_channel_sigma
    na127_peak_channel_hi = na127_peak_channel + A * na127_peak_channel_sigma
    na127_peak_energy_lo = fitfunc(pf, na127_peak_channel_lo)
    na127_peak_energy_hi = fitfunc(pf, na127_peak_channel_hi)
    na127_peak_theta_max = scatter_angle(na127_peak_energy_hi, na127_peak_energy_lo)
    na127_peak_theta_max_deg = na127_peak_theta_max * 180 / np.pi
    print("e_l: {}, e_c: {}, e_h: {}, theta_max: {}, A: {}"
          "".format(na127_peak_energy_lo, NA22_127_ENERGY,
                    na127_peak_energy_hi, na127_peak_theta_max_deg, A))
    return


def test():
    scatter_energy_expected = na127_peak_energy * 0.9
    scatter_angle_ = scatter_angle(na127_peak_energy, scatter_energy_expected)
    scatter_energy_ = scatter_energy(na127_peak_energy, scatter_angle_)
    assert(np.allclose(scatter_energy_expected, scatter_energy_))
    return
