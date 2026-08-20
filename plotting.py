import matplotlib.pyplot as plt
import numpy as np


def plot_magnetisation_components(t_series, Mx_series, My_series, Mz_series, M_series):
    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, Mx_series, label=r"$M_x$")
    plt.plot(t_series * 1e12, My_series, label=r"$M_y$")
    plt.plot(t_series * 1e12, Mz_series, label=r"$M_z$")
    plt.plot(t_series * 1e12, M_series, label=r"$|\mathbf{M}|$", linewidth=1.5)

    plt.xlabel("Time (ps)")
    plt.ylabel("Magnetisation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_magnetisation_magnitude(t_series, M_series):
    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, M_series)

    plt.xlabel("Time (ps)")
    plt.ylabel(r"$|M|$")
    plt.tight_layout()


def plot_transverse_magnetisation(t_series, Mx_series, My_series):
    Mxy_series = np.sqrt(Mx_series * Mx_series + My_series * My_series)

    plt.figure(figsize=(7, 4), dpi=150)

    plt.plot(t_series * 1e12, Mxy_series)

    plt.xlabel("Time (ps)")
    plt.ylabel(r"$M_\perp$")
    plt.tight_layout()
