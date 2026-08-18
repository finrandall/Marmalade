import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch


def build_fm_transverse_field(Sx_samp, Sy_samp):
    return Sx_samp + 1j * Sy_samp


def build_afm_sublattice_transverse_fields(Sx_samp, Sy_samp, Lx, Ly, Lz):
    nsamp = Sx_samp.shape[0]

    Sx_zyx = Sx_samp.reshape(nsamp, Lz, Ly, Lx)
    Sy_zyx = Sy_samp.reshape(nsamp, Lz, Ly, Lx)

    phi_A = np.zeros((nsamp, Lz, Ly, Lx), dtype=np.complex64)
    phi_B = np.zeros((nsamp, Lz, Ly, Lx), dtype=np.complex64)

    for iz in range(Lz):
        for iy in range(Ly):
            for ix in range(Lx):
                if (ix + iy + iz) % 2 == 0:
                    phi_A[:, iz, iy, ix] = Sx_zyx[:, iz, iy, ix] + 1j * Sy_zyx[:, iz, iy, ix]
                else:
                    phi_B[:, iz, iy, ix] = Sx_zyx[:, iz, iy, ix] - 1j * Sy_zyx[:, iz, iy, ix]

    return phi_A, phi_B


def compute_full_spectrum(phi, Lx, Ly, Lz, nsamp, dt_sample):
    fs = 1.0 / dt_sample

    nperseg = max(8, nsamp // 8)
    nperseg = min(nperseg, nsamp)
    noverlap = nperseg // 2

    if phi.ndim == 2:
        phi_zyx = phi.reshape(nsamp, Lz, Ly, Lx)
    else:
        phi_zyx = phi

    phi_k = np.fft.fftn(phi_zyx, axes=(1, 2, 3), norm="ortho")
    phi_k = phi_k - phi_k.mean(axis=0, keepdims=True)

    kx = 2.0 * np.pi * np.fft.fftfreq(Lx, d=1.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ly, d=1.0)
    kz = 2.0 * np.pi * np.fft.fftfreq(Lz, d=1.0)

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)

    f, Pxx = welch(phi_k, fs=fs, axis=0, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False, return_onesided=False, scaling="spectrum")

    omega = 2.0 * np.pi * f
    omega = np.fft.fftshift(omega)

    S_kw = np.fft.fftshift(Pxx, axes=(0,))
    S_kw = np.fft.fftshift(S_kw, axes=(1, 2, 3))

    return kx, ky, kz, omega, S_kw


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
    path_kz = np.zeros_like(kx)
    path_dist = np.linspace(-np.pi, np.pi, len(kx))

    tick_positions = [-np.pi, 0.0, np.pi]
    tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]

    return path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels


def build_high_symmetry_path():
    points_per_segment = 128

    gamma_point = np.array([0.0, 0.0, 0.0])
    x_point = np.array([np.pi, 0.0, 0.0])
    m_point = np.array([np.pi, np.pi, 0.0])

    sym_points = [gamma_point, x_point, m_point, gamma_point]
    tick_labels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]

    path_kx = []
    path_ky = []
    path_kz = []
    path_dist = []

    tick_positions = [0.0]
    dist = 0.0

    for p0, p1 in zip(sym_points[:-1], sym_points[1:]):
        segment = np.linspace(p0, p1, points_per_segment, endpoint=False)

        for k in segment:
            if len(path_dist) > 0:
                previous_k = np.array([path_kx[-1], path_ky[-1], path_kz[-1]])
                dist += np.linalg.norm(k - previous_k)

            path_kx.append(k[0])
            path_ky.append(k[1])
            path_kz.append(k[2])
            path_dist.append(dist)

        tick_positions.append(dist + np.linalg.norm(p1 - segment[-1]))

    path_kx.append(sym_points[-1][0])
    path_ky.append(sym_points[-1][1])
    path_kz.append(sym_points[-1][2])
    path_dist.append(tick_positions[-1])

    return np.array(path_kx), np.array(path_ky), np.array(path_kz), np.array(path_dist), tick_positions, tick_labels


def build_path(path_mode, kx=None):
    if path_mode == "kx":
        if kx is None:
            path_kx = np.linspace(-np.pi, np.pi, 128)
            path_ky = np.zeros_like(path_kx)
            path_kz = np.zeros_like(path_kx)
            path_dist = np.linspace(-np.pi, np.pi, 128)
            tick_positions = [-np.pi, 0.0, np.pi]
            tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]
        else:
            path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels = build_kx_path(kx)

    elif path_mode == "high_symmetry":
        path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels = build_high_symmetry_path()

    else:
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    return path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels


def extract_spectrum_path(S_kw, kx, ky, kz, omega, path_kx, path_ky, path_kz):
    S_path = np.empty((len(omega), len(path_kx)), dtype=np.float64)

    for n in range(len(path_kx)):
        ix = nearest_index(kx, path_kx[n])
        iy = nearest_index(ky, path_ky[n])
        iz = nearest_index(kz, path_kz[n])
        S_path[:, n] = S_kw[:, iz, iy, ix]

    return S_path


def fm_analytic_dispersion(kx, ky, kz, Lz, J, K, h, gamma, mu):
    if Lz == 1:
        gamma_1 = 0.5 * (np.cos(kx) + np.cos(ky))
        exchange = 4.0 * J * (1.0 - gamma_1)
    else:
        gamma_1 = (np.cos(kx) + np.cos(ky) + np.cos(kz)) / 3.0
        exchange = 6.0 * J * (1.0 - gamma_1)

    omega = (gamma / mu) * (exchange + K + h)

    return omega


def afm_analytic_branches(kx, ky, kz, Lz, J, K, h, S_spin, gamma, mu):
    if Lz == 1:
        z_coord = 4.0
        gamma_1 = 0.5 * (np.cos(kx) + np.cos(ky))
    else:
        z_coord = 6.0
        gamma_1 = (np.cos(kx) + np.cos(ky) + np.cos(kz)) / 3.0

    J_scale = -J

    if J_scale <= 0.0:
        raise ValueError("For AFM analytic dispersion, J must be negative with this Hamiltonian convention.")

    JSz = J_scale * S_spin * z_coord

    lambda_sw = 1.0 + K / JSz
    eta = h / JSz

    root_arg = lambda_sw * lambda_sw - gamma_1 * gamma_1
    root_arg = np.maximum(root_arg, 0.0)

    root = np.sqrt(root_arg)

    omega_plus = (gamma / mu) * JSz * (root + eta)
    omega_minus = (gamma / mu) * JSz * (root - eta)

    return omega_plus, omega_minus


def plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves=None):
    omega_min = 0.0

    if analytic_curves is None:
        omega_max = np.percentile(omega[omega > 0.0], 90.0)
    else:
        omega_max = 1.1 * max(np.max(np.abs(curve_omega)) for _, curve_omega in analytic_curves)

    mask = (omega >= omega_min) & (omega <= omega_max)

    omega_cut = omega[mask]
    S_cut = S_path[mask, :]

    intensity = np.log(S_cut + 1e-30)

    x_edges = centres_to_edges(path_dist)
    y_edges = centres_to_edges(omega_cut)

    plt.figure(figsize=(5, 3.5), dpi=120)
    plt.pcolormesh(x_edges, y_edges, intensity, shading="auto", cmap="PuBu_r")

    if analytic_curves is not None:
        analytic_colors = ["red", "orange"]

        for n, (curve_dist, curve_omega) in enumerate(analytic_curves):
            color = analytic_colors[n] if n < len(analytic_colors) else "red"
            plt.plot(curve_dist, np.abs(curve_omega), linewidth=0.7, alpha=0.8, linestyle="-", color=color)

    for tick in tick_positions:
        plt.axvline(tick, linewidth=0.5, alpha=0.4)

    plt.xlim(tick_positions[0], tick_positions[-1])
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel(r"$\omega$")
    plt.tight_layout()


def plot_fm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, Lz, nsamp, dt_sample, J, K, h, gamma, mu, path_mode="kx", show_analytic=True):
    if path_mode not in ("kx", "high_symmetry"):
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    phi = build_fm_transverse_field(Sx_samp, Sy_samp)

    kx, ky, kz, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, Lz, nsamp, dt_sample)

    path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels = build_path(path_mode, kx=kx)

    S_path = extract_spectrum_path(S_kw, kx, ky, kz, omega, path_kx, path_ky, path_kz)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_kz, analytic_path_dist, _, _ = build_path(path_mode, kx=None)
        analytic = fm_analytic_dispersion(analytic_path_kx, analytic_path_ky, analytic_path_kz, Lz, J, K, h, gamma, mu)
        analytic_curves = [(analytic_path_dist, analytic)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)


def plot_afm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, Lz, nsamp, dt_sample, J, K, h, S_spin, gamma, mu, path_mode="kx", show_analytic=True, branch_mode="sum"):
    if path_mode not in ("kx", "high_symmetry"):
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    phi_A, phi_B = build_afm_sublattice_transverse_fields(Sx_samp, Sy_samp, Lx, Ly, Lz)

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

    kx, ky, kz, omega, S_kw = compute_full_spectrum(phi, Lx, Ly, Lz, nsamp, dt_sample)

    path_kx, path_ky, path_kz, path_dist, tick_positions, tick_labels = build_path(path_mode, kx=kx)

    S_path = extract_spectrum_path(S_kw, kx, ky, kz, omega, path_kx, path_ky, path_kz)

    if show_analytic:
        analytic_path_kx, analytic_path_ky, analytic_path_kz, analytic_path_dist, _, _ = build_path(path_mode, kx=None)
        omega_plus, omega_minus = afm_analytic_branches(analytic_path_kx, analytic_path_ky, analytic_path_kz, Lz, J, K, h, S_spin, gamma, mu)
        analytic_curves = [(analytic_path_dist, omega_plus), (analytic_path_dist, omega_minus)]
    else:
        analytic_curves = None

    plot_spectrum_path(S_path, omega, path_dist, tick_positions, tick_labels, analytic_curves)
