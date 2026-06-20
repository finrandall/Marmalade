import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch


def build_fm_transverse_field(Sx_samp, Sy_samp):
    return Sx_samp + 1j * Sy_samp


def build_afm_sublattice_transverse_fields(Sx_samp, Sy_samp, Lx, Ly):
    nsamp = Sx_samp.shape[0]

    Sx_yx = Sx_samp.reshape(nsamp, Ly, Lx)
    Sy_yx = Sy_samp.reshape(nsamp, Ly, Lx)

    phi_A = np.zeros((nsamp, Ly, Lx), dtype=np.complex64)
    phi_B = np.zeros((nsamp, Ly, Lx), dtype=np.complex64)

    for iy in range(Ly):
        for ix in range(Lx):
            if (ix + iy) % 2 == 0:
                phi_A[:, iy, ix] = Sx_yx[:, iy, ix] + 1j * Sy_yx[:, iy, ix]
            else:
                phi_B[:, iy, ix] = Sx_yx[:, iy, ix] - 1j * Sy_yx[:, iy, ix]

    return phi_A, phi_B


def compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample):
    fs = 1.0 / dt_sample

    nperseg = max(8, nsamp // 8)
    noverlap = nperseg // 2

    if phi.ndim == 2:
        phi_yx = phi.reshape(nsamp, Ly, Lx)
    else:
        phi_yx = phi

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


def centres_to_edges(centres):
    centres = np.asarray(centres, dtype=np.float64)

    if len(centres) < 2:
        return np.array([centres[0] - 0.5, centres[0] + 0.5], dtype=np.float64)

    edges = np.empty(len(centres) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])

    return edges


def build_kx_path(kx):
    path_kx = kx
    path_ky = np.zeros_like(kx)
    path_dist = np.linspace(-np.pi, np.pi, len(kx))

    tick_positions = [-np.pi, 0.0, np.pi]
    tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]

    return path_kx, path_ky, path_dist, tick_positions, tick_labels


def build_high_symmetry_path():
    points_per_segment = 128

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

    for p0, p1 in zip(sym_points[:-1], sym_points[1:]):
        segment = np.linspace(p0, p1, points_per_segment, endpoint=False)

        for k in segment:
            if len(path_dist) > 0:
                previous_k = np.array([path_kx[-1], path_ky[-1]])
                dist += np.linalg.norm(k - previous_k)

            path_kx.append(k[0])
            path_ky.append(k[1])
            path_dist.append(dist)

        tick_positions.append(dist + np.linalg.norm(p1 - segment[-1]))

    path_kx.append(sym_points[-1][0])
    path_ky.append(sym_points[-1][1])
    path_dist.append(tick_positions[-1])

    return np.array(path_kx), np.array(path_ky), np.array(path_dist), tick_positions, tick_labels


def build_path(path_mode, kx=None):
    if path_mode == "kx":
        if kx is None:
            path_kx = np.linspace(-np.pi, np.pi, 128)
            path_ky = np.zeros_like(path_kx)
            path_dist = np.linspace(-np.pi, np.pi, 128)
            tick_positions = [-np.pi, 0.0, np.pi]
            tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]
        else:
            path_kx, path_ky, path_dist, tick_positions, tick_labels = build_kx_path(kx)

    elif path_mode == "high_symmetry":
        path_kx, path_ky, path_dist, tick_positions, tick_labels = build_high_symmetry_path()

    else:
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    return path_kx, path_ky, path_dist, tick_positions, tick_labels


def extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky):
    S_path = np.empty((len(omega), len(path_kx)), dtype=np.float64)

    for n in range(len(path_kx)):
        ix = nearest_index(kx, path_kx[n])
        iy = nearest_index(ky, path_ky[n])
        S_path[:, n] = S_kw[:, iy, ix]

    return S_path


def fm_analytic_dispersion(kx, ky, J1, J2, K, h, gamma, mu):
    gamma_1 = 0.5 * (np.cos(kx) + np.cos(ky))
    gamma_2 = np.cos(kx) * np.cos(ky)

    omega = (gamma / mu) * (4.0 * J1 * (1.0 - gamma_1) + 4.0 * J2 * (1.0 - gamma_2) + K + h)

    return omega


def afm_analytic_branches(kx, ky, J1, J2, K, h, S_spin, gamma, mu):
    z_coord = 4.0

    J1_scale = -J1
    J2_scale = -J2

    if J1_scale <= 0.0:
        raise ValueError("For AFM analytic dispersion, J1 must be negative with this Hamiltonian convention.")

    JSz = J1_scale * S_spin * z_coord

    gamma_1 = 0.5 * (np.cos(kx) + np.cos(ky))
    gamma_2 = np.cos(kx) * np.cos(ky)

    alpha = J2_scale / J1_scale
    lambda_sw = 1.0 + K / JSz - alpha * (1.0 - gamma_2)
    eta = h / JSz

    root_arg = lambda_sw * lambda_sw - gamma_1 * gamma_1
    root_arg = np.maximum(root_arg, 0.0)

    root = np.sqrt(root_arg)

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

    x_edges = centres_to_edges(path_dist)
    y_edges = centres_to_edges(omega_cut)

    plt.figure(figsize=(5, 3.5), dpi=120)
    plt.pcolormesh(x_edges, y_edges, intensity, shading="auto", cmap="PuBu_r")

    if analytic_curves is not None:
        for curve_dist, curve_omega in analytic_curves:
            plt.plot(curve_dist, curve_omega, linewidth=0.7, alpha=0.8, linestyle="-", color="red")

    for tick in tick_positions:
        plt.axvline(tick, linewidth=0.5, alpha=0.4)

    plt.xlim(tick_positions[0], tick_positions[-1])
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel(r"$\omega$")
    plt.tight_layout()


def plot_fm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J1, J2, K, h, gamma, mu, path_mode="kx", show_analytic=True):
    if path_mode not in ("kx", "high_symmetry"):
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    phi = build_fm_transverse_field(Sx_samp, Sy_samp)

    kx, ky, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    path_kx, path_ky, path_dist, tick_positions, tick_labels = build_path(path_mode, kx=kx)

    S_path = extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_dist, _, _ = build_path(path_mode, kx=None)
        analytic = fm_analytic_dispersion(analytic_path_kx, analytic_path_ky, J1, J2, K, h, gamma, mu)
        analytic_curves = [(analytic_path_dist, analytic)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)


def plot_afm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J1, J2, K, h, S_spin, gamma, mu, path_mode="kx", show_analytic=True, branch_mode="sum"):
    if path_mode not in ("kx", "high_symmetry"):
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    phi_A, phi_B = build_afm_sublattice_transverse_fields(Sx_samp, Sy_samp, Lx, Ly)

    if branch_mode == "sum":
        phi = phi_A + phi_B
    elif branch_mode == "difference":
        phi = phi_A - phi_B
    elif branch_mode == "A":
        phi = phi_A
    elif branch_mode == "B":
        phi = phi_B
    else:
        raise ValueError("branch_mode must be 'sum', 'difference', 'A', or 'B'")

    kx, ky, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, nsamp, dt_sample)

    path_kx, path_ky, path_dist, tick_positions, tick_labels = build_path(path_mode, kx=kx)

    S_path = extract_spectrum_path(S_kw, kx, ky, omega, path_kx, path_ky)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_dist, _, _ = build_path(path_mode, kx=None)
        omega_plus, omega_minus = afm_analytic_branches(analytic_path_kx, analytic_path_ky, J1, J2, K, h, S_spin, gamma, mu)
        analytic_curves = [(analytic_path_dist, omega_plus), (analytic_path_dist, omega_minus)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)