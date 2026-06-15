import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch


def build_fm_transverse_field(Sx_samp, Sy_samp):
    return Sx_samp + 1j * Sy_samp


def build_afm_local_transverse_field(Sx_samp, Sy_samp, Lx, Ly):
    n_sites = Lx * Ly
    parity = np.empty(n_sites, dtype=np.int32)

    for iy in range(Ly):
        for ix in range(Lx):
            i = ix + Lx * iy
            parity[i] = (ix + iy) % 2

    phi = np.empty_like(Sx_samp, dtype=np.complex64)

    phi[:, parity == 0] = Sx_samp[:, parity == 0] + 1j * Sy_samp[:, parity == 0]
    phi[:, parity == 1] = Sx_samp[:, parity == 1] - 1j * Sy_samp[:, parity == 1]

    return phi


def compute_kx_spectrum(phi, Lx, Ly, nsamp, dt_sample):
    fs = 1.0 / dt_sample

    nperseg = max(8, nsamp // 8)
    noverlap = nperseg // 2

    phi_yx = phi.reshape(nsamp, Ly, Lx)

    phi_k = np.fft.fftn(phi_yx, axes=(1, 2), norm="ortho")
    phi_k = phi_k - phi_k.mean(axis=0, keepdims=True)

    kx = 2.0 * np.pi * np.fft.fftfreq(Lx, d=1.0)
    kx = np.fft.fftshift(kx)

    f, Pxx = welch(phi_k, fs=fs, axis=0, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False, return_onesided=False, scaling="spectrum")

    omega = 2.0 * np.pi * f
    omega = np.fft.fftshift(omega)

    S_kw = np.fft.fftshift(Pxx, axes=(0,))
    S_kw = np.fft.fftshift(S_kw, axes=(1, 2))

    ky_index = Ly // 2
    S_omega_kx = S_kw[:, ky_index, :]

    return kx, omega, S_omega_kx


def fm_analytic_dispersion(kx, J, K, gamma, mu):
    gamma_k = 0.5 * (np.cos(kx) + 1.0)

    omega = (gamma / mu) * (4.0 * J * (1.0 - gamma_k) + 2.0 * K)

    return omega


def afm_analytic_branches(kx, ky, J, kappa_z, h, S_spin, gamma, mu):
    z_coord = 4.0
    JSz = J * S_spin * z_coord

    gamma_k = 0.5 * (np.cos(kx) + np.cos(ky))
    lambda_sw = 1.0 + kappa_z / JSz
    eta = h / JSz

    root = np.sqrt(lambda_sw * lambda_sw - gamma_k * gamma_k)

    omega_plus = (gamma / mu) * JSz * (root + eta)
    omega_minus = (gamma / mu) * JSz * (root - eta)

    return omega_plus, omega_minus


def plot_fm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J, K, gamma, mu, show_analytic=True):
    phi = build_fm_transverse_field(Sx_samp, Sy_samp)

    kx, omega, S_omega_kx = compute_kx_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    analytic = fm_analytic_dispersion(kx, J, K, gamma, mu)

    omega_min = -0.05 * np.max(analytic)
    omega_max = 1.1 * np.max(analytic)

    mask = (omega >= omega_min) & (omega <= omega_max)

    omega_cut = omega[mask]
    S_cut = S_omega_kx[mask, :]

    intensity = np.log(S_cut + 1e-30)

    plt.figure(figsize=(5, 4), dpi=200)
    plt.imshow(intensity, origin="lower", aspect="auto", cmap="PuBu_r", extent=[kx[0], kx[-1], omega_cut[0], omega_cut[-1]])

    if show_analytic:
        plt.plot(kx, analytic, color="red", linewidth=0.7, alpha=0.8)

    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$\omega$")
    plt.tight_layout()


def plot_afm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J, kappa_z, h, S_spin, gamma, mu, show_analytic=True):
    phi = build_afm_local_transverse_field(Sx_samp, Sy_samp, Lx, Ly)

    kx, omega, S_omega_kx = compute_kx_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    ky = np.zeros_like(kx)
    omega_plus, omega_minus = afm_analytic_branches(kx, ky, J, kappa_z, h, S_spin, gamma, mu)

    omega_min = 0.0
    omega_max = 1.1 * max(np.max(omega_plus), np.max(omega_minus))

    mask = (omega >= omega_min) & (omega <= omega_max)

    omega_cut = omega[mask]
    S_cut = S_omega_kx[mask, :]

    intensity = np.log(S_cut + 1e-30)

    plt.figure(figsize=(7, 5), dpi=200)
    plt.imshow(intensity, origin="lower", aspect="auto", cmap="PuBu_r", extent=[kx[0], kx[-1], omega_cut[0], omega_cut[-1]])

    if show_analytic:
        plt.plot(kx, omega_plus, color="red", linewidth=0.7, alpha=0.8)
        plt.plot(kx, omega_minus, color="orange", linewidth=0.7, alpha=0.8, linestyle="--")

    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$\omega$")
    plt.tight_layout()