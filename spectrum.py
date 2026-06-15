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


def compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample):
    fs = 1.0 / dt_sample

    nperseg = max(8, nsamp // 8)
    noverlap = nperseg // 2

    phi_yx = phi.reshape(nsamp, Ly, Lx)

    phi_k = np.fft.fftn(phi_yx, axes=(1, 2), norm="ortho")
    phi_k = phi_k - phi_k.mean(axis=0, keepdims=True)

    kx = 2.0 * np.pi * np.fft.fftfreq(Lx, d=1.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ly, d=1.0)

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    f, Pxx = welch(phi_k, fs=fs, axis=0, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False, return_onesided=False, scaling="spectrum")

    omega = 2.0 * np.pi * f
    omega = np.fft.fftshift(omega)

    S_kw = np.fft.fftshift(Pxx, axes=(0,))
    S_kw = np.fft.fftshift(S_kw, axes=(1, 2))

    return kx, ky, omega, S_kw


def nearest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def build_kx_path(kx):
    path_kx = kx
    path_ky = np.zeros_like(kx)
    path_dist = kx.copy()
    tick_positions = [kx[0], 0.0, kx[-1]]
    tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]

    return path_kx, path_ky, path_dist, tick_positions, tick_labels


def build_high_symmetry_path(points_per_segment=128):
    gamma_point = np.array([0.0, 0.0])
    x_point = np.array([np.pi, 0.0])
    m_point = np.array([np.pi, np.pi])

    sym_points = [gamma_point, x_point, m_point, gamma_point]
    tick_labels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]

    path_kx = []
    path_ky = []
    path_dist = []

    tick_positions = [0.0]

    dist = 0.0

    for segment_index, (p0, p1) in enumerate(zip(sym_points[:-1], sym_points[1:])):
        segment = np.linspace(p0, p1, points_per_segment, endpoint=False)

        if segment_index == 0:
            previous_k = segment[0]
        else:
            previous_k = sym_points[segment_index]

        for k in segment:
            if len(path_dist) > 0:
                dist += np.linalg.norm(k - previous_k)

            path_kx.append(k[0])
            path_ky.append(k[1])
            path_dist.append(dist)

            previous_k = k

        segment_length = np.linalg.norm(p1 - p0)
        tick_positions.append(tick_positions[-1] + segment_length)

    path_kx.append(sym_points[-1][0])
    path_ky.append(sym_points[-1][1])
    path_dist.append(tick_positions[-1])

    return np.array(path_kx), np.array(path_ky), np.array(path_dist), tick_positions, tick_labels


def extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky):
    S_path = np.empty((len(omega), len(path_kx)), dtype=np.float64)

    for n in range(len(path_kx)):
        ix = nearest_index(kx, path_kx[n])
        iy = nearest_index(ky, path_ky[n])
        S_path[:, n] = S_kw[:, iy, ix]

    return S_path


def fm_analytic_dispersion(kx, ky, J, K, gamma, mu):
    gamma_k = 0.5 * (np.cos(kx) + np.cos(ky))

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


def plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves=None):
    if analytic_curves is None:
        omega_max = np.percentile(omega[omega > 0.0], 90.0)
    else:
        omega_max = 1.1 * max(np.max(curve_omega) for _, curve_omega in analytic_curves)

    omega_min = 0.0
    mask = (omega >= omega_min) & (omega <= omega_max)

    omega_cut = omega[mask]
    S_cut = S_path[mask, :]

    intensity = np.log(S_cut + 1e-30)

    x_grid, y_grid = np.meshgrid(path_dist, omega_cut)

    plt.figure(figsize=(5, 3.5), dpi=120)
    plt.pcolormesh(x_grid, y_grid, intensity, shading="auto", cmap="PuBu_r")

    if analytic_curves is not None:
        for curve_dist, curve_omega in analytic_curves:
            plt.plot(curve_dist, curve_omega, linewidth=0.7, alpha=0.8, linestyle="-", color="red")

    plt.xticks(tick_positions, tick_labels)
    plt.ylabel(r"$\omega$")
    plt.tight_layout()

def build_analytic_path(path_mode, points_per_segment=1024):
    if path_mode == "kx":
        path_kx = np.linspace(-np.pi, np.pi, points_per_segment)
        path_ky = np.zeros_like(path_kx)
        path_dist = path_kx.copy()
        tick_positions = [-np.pi, 0.0, np.pi]
        tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]

    elif path_mode == "high_symmetry":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_high_symmetry_path(points_per_segment)

    else:
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    return path_kx, path_ky, path_dist, tick_positions, tick_labels


def plot_fm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J, K, gamma, mu, path_mode="kx", points_per_segment=128, show_analytic=True):
    phi = build_fm_transverse_field(Sx_samp, Sy_samp)

    kx, ky, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    if path_mode == "kx":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_kx_path(kx)

    elif path_mode == "high_symmetry":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_high_symmetry_path(points_per_segment)

    else:
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    S_path = extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_dist, _, _ = build_analytic_path(path_mode, points_per_segment=1024)
        analytic = fm_analytic_dispersion(analytic_path_kx, analytic_path_ky, J, K, gamma, mu)
        analytic_curves = [(analytic_path_dist, analytic)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)


def plot_afm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J, kappa_z, h, S_spin, gamma, mu, path_mode="kx", points_per_segment=128, show_analytic=True):
    phi = build_afm_local_transverse_field(Sx_samp, Sy_samp, Lx, Ly)

    kx, ky, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    if path_mode == "kx":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_kx_path(kx)
        path_ky = np.zeros_like(path_kx)

    elif path_mode == "high_symmetry":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_high_symmetry_path(points_per_segment)

    else:
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    S_path = extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_dist, _, _ = build_analytic_path(path_mode, points_per_segment=1024)
        omega_plus, omega_minus = afm_analytic_branches(analytic_path_kx, analytic_path_ky, J, kappa_z, h, S_spin, gamma, mu)
        analytic_curves = [(analytic_path_dist, omega_plus), (analytic_path_dist, omega_minus)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)