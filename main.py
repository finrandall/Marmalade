import time

from interface import get_parameters

import matplotlib.pyplot as plt
import numpy as np

from initialise import init_spins
from lattice import build_neighbour_list

from noise import init_quantum_noise
from noise import quantum_noise_amplitudes

from evolve import evolve_quantum_time_series
from evolve import evolve_classical_time_series
from evolve import evolve_deterministic_time_series
from evolve import evolve_quantum_spin_samples
from evolve import evolve_classical_spin_samples
from evolve import evolve_deterministic_spin_samples

from magnetisation import plot_magnetisation_magnitude
from magnetisation import plot_transverse_magnetisation

from spectrum import plot_fm_magnon_spectrum
from spectrum import plot_afm_magnon_spectrum

params = get_parameters()

output_mode = params["output_mode"]
noise_mode = params["noise_mode"]

spectrum_path = params["spectrum_path"]
show_analytic = params["show_analytic"]

initial_state = params["initial_state"]

T = params["T"]

Lx = params["Lx"]
Ly = params["Ly"]
N = Lx * Ly

dt = params["dt"]
end_time = params["end_time"]
burn_in_time = params["burn_in_time"]
iterations = int(end_time / dt)

stride = params["stride"]

J1 = params["J1"]
J2 = params["J2"]
K = params["K"]
h = params["h"]

mu = 9.274e-24
gamma = 1.7609e11
lam = params["lam"]
hbar = 1.054571817e-34
kB = 1.380649e-23
eV = 1.602176634e-19

J1 *= eV
J2 *= eV
K *= eV
h *= eV

J1_mu = J1 / mu
J2_mu = J2 / mu
K_mu = K / mu
h_mu = h / mu

kT = kB * T

Gamma = lam * kT / ((1.0 + lam * lam) * gamma * mu)

dt_q = dt * kT / hbar

q_pref = np.sqrt(2.0 * Gamma * kT / hbar)
c_pref = np.sqrt(2.0 * Gamma / dt)

amp5, amp6 = quantum_noise_amplitudes(dt_q)

S = init_spins(Lx, Ly, mode=initial_state)
nn, nnn = build_neighbour_list(Lx, Ly)

print("Running simulation...")

t0 = time.time()

if output_mode == "magnetisation":
    n_samples = (iterations + stride - 1) // stride

    t_series = np.empty(n_samples, dtype=np.float64)
    Mx_series = np.empty(n_samples, dtype=np.float64)
    My_series = np.empty(n_samples, dtype=np.float64)
    Mz_series = np.empty(n_samples, dtype=np.float64)
    M_series = np.empty(n_samples, dtype=np.float64)

    if noise_mode == "quantum":
        z5, v5, z6, v6 = init_quantum_noise(N)
        evolve_quantum_time_series(S, nn, nnn, z5, v5, z6, v6, t_series, Mx_series, My_series, Mz_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, q_pref, dt_q, amp5, amp6, iterations, stride)

    elif noise_mode == "classical":
        evolve_classical_time_series(S, nn, nnn, t_series, Mx_series, My_series, Mz_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, c_pref, iterations, stride)

    elif noise_mode == "none":
        evolve_deterministic_time_series(S, nn, nnn, t_series, Mx_series, My_series, Mz_series, M_series, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, iterations, stride)

    else:
        raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

    plot_magnetisation_magnitude(t_series, M_series)
    plot_transverse_magnetisation(t_series, Mx_series, My_series)

    t1 = time.time()
    print(f"Simulation time: {t1 - t0:.3f} s")

    plt.show()

elif output_mode == "spectrum":
    burn_in_steps = int(burn_in_time / dt)

    if burn_in_steps >= iterations:
        raise ValueError("burn_in_time must be smaller than end_time")

    rem = iterations - burn_in_steps
    nsamp = (rem + stride - 1) // stride
    dt_sample = stride * dt

    Sx_samp = np.empty((nsamp, N), dtype=np.float32)
    Sy_samp = np.empty((nsamp, N), dtype=np.float32)
    Sz_samp = np.empty((nsamp, N), dtype=np.float32)

    if noise_mode == "quantum":
        z5, v5, z6, v6 = init_quantum_noise(N)
        evolve_quantum_spin_samples(S, nn, nnn, z5, v5, z6, v6, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, q_pref, dt_q, amp5, amp6, iterations, burn_in_steps, stride)

    elif noise_mode == "classical":
        evolve_classical_spin_samples(S, nn, nnn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, c_pref, iterations, burn_in_steps, stride)

    elif noise_mode == "none":
        evolve_deterministic_spin_samples(S, nn, nnn, Sx_samp, Sy_samp, Sz_samp, dt, gamma, lam, J1_mu, J2_mu, K_mu, h_mu, iterations, burn_in_steps, stride)

    else:
        raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

    if spectrum_path not in ("kx", "high_symmetry"):
        raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

    if initial_state == "fm":
        plot_fm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J1, J2, K, h, gamma, mu, path_mode=spectrum_path, show_analytic=show_analytic)

    elif initial_state == "afm":
        plot_afm_magnon_spectrum(Sx_samp, Sy_samp, Lx, Ly, nsamp, dt_sample, J1, J2, K, h, 1.0, gamma, mu, path_mode=spectrum_path, show_analytic=show_analytic)

    else:
        raise ValueError("initial_state must be 'fm' or 'afm'")

    t1 = time.time()
    print(f"Simulation time: {t1 - t0:.3f} s")

    plt.show()

else:
    raise ValueError("output_mode must be 'magnetisation' or 'spectrum'")