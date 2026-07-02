import time

import matplotlib.pyplot as plt
import numpy as np

from initialise import init_spins
from lattice import build_neighbour_list

from noise import init_quantum_noise
from noise import quantum_noise_amplitudes

from evolve import evolve_quantum_time_series
from evolve import evolve_quantum_spin_samples
from evolve import evolve_classical_spin_samples

from evolve import evolve_classical_time_series
from evolve import evolve_classical_rkmk2_time_series

from evolve import evolve_deterministic_heun_time_series
from evolve import evolve_deterministic_rkmk2_time_series
from evolve import evolve_deterministic_spin_samples

from magnetisation import plot_magnetisation_magnitude
from magnetisation import plot_transverse_magnetisation

from spectrum import plot_fm_magnon_spectrum
from spectrum import plot_afm_magnon_spectrum


class Simulation:
    def __init__(self, params):
        self.params = params

        self.output_mode = params["output_mode"]
        self.noise_mode = params["noise_mode"]
        self.integrator = params["integrator"]

        self.spectrum_path = params["spectrum_path"]
        self.show_analytic = params["show_analytic"]
        self.initial_state = params["initial_state"]

        self.T = params["T"]

        self.Lx = params["Lx"]
        self.Ly = params["Ly"]
        self.Lz = params["Lz"]
        self.pbc_x = params["pbc_x"]
        self.pbc_y = params["pbc_y"]
        self.pbc_z = params["pbc_z"]
        self.N = self.Lx * self.Ly * self.Lz

        self.dt = params["dt"]
        self.end_time = params["end_time"]
        self.burn_in_time = params["burn_in_time"]
        self.iterations = int(self.end_time / self.dt)

        self.stride = params["stride"]

        self.J1_input = params["J1"]
        self.J2_input = params["J2"]
        self.K_input = params["K"]
        self.h_input = params["h"]

        self.mu = 9.274e-24
        self.gamma = 1.7609e11
        self.lam = params["lam"]
        self.hbar = 1.054571817e-34
        self.kB = 1.380649e-23
        self.eV = 1.602176634e-19

        self.J1 = self.J1_input * self.eV
        self.J2 = self.J2_input * self.eV
        self.K = self.K_input * self.eV
        self.h = self.h_input * self.eV

        self.J1_mu = self.J1 / self.mu
        self.J2_mu = self.J2 / self.mu
        self.K_mu = self.K / self.mu
        self.h_mu = self.h / self.mu

        self.kT = self.kB * self.T
        self.Gamma = self.lam * self.kT / ((1.0 + self.lam * self.lam) * self.gamma * self.mu)
        self.dt_q = self.dt * self.kT / self.hbar

        self.q_pref = np.sqrt(2.0 * self.Gamma * self.kT / self.hbar)
        self.c_pref = np.sqrt(2.0 * self.Gamma / self.dt)

        self.amp5, self.amp6 = quantum_noise_amplitudes(self.dt_q)

        self.S = None
        self.nn = None
        self.nnn = None

    def initialise(self):
        self.S = init_spins(self.Lx, self.Ly, self.Lz, mode=self.initial_state)
        self.nn, self.nnn = build_neighbour_list(self.Lx, self.Ly, self.Lz, self.pbc_x, self.pbc_y, self.pbc_z)

    def run(self):
        self.initialise()

        print("Running simulation...")

        if self.output_mode == "magnetisation":
            self.run_magnetisation()
        elif self.output_mode == "spectrum":
            self.run_spectrum()
        else:
            raise ValueError("output_mode must be 'magnetisation' or 'spectrum'")

    def run_magnetisation(self):
        t0 = time.time()

        n_samples = (self.iterations + self.stride - 1) // self.stride

        t_series = np.empty(n_samples, dtype=np.float64)
        Mx_series = np.empty(n_samples, dtype=np.float64)
        My_series = np.empty(n_samples, dtype=np.float64)
        Mz_series = np.empty(n_samples, dtype=np.float64)
        M_series = np.empty(n_samples, dtype=np.float64)

        if self.noise_mode == "quantum":
            self.run_quantum_time_series(t_series, Mx_series, My_series, Mz_series, M_series)

        elif self.noise_mode == "classical":
            self.run_classical_time_series(t_series, Mx_series, My_series, Mz_series, M_series)

        elif self.noise_mode == "none":
            self.run_deterministic_time_series(t_series, Mx_series, My_series, Mz_series, M_series)

        else:
            raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

        plot_magnetisation_magnitude(t_series, M_series)
        plot_transverse_magnetisation(t_series, Mx_series, My_series)

        t1 = time.time()
        print(f"Simulation time: {t1 - t0:.3f} s")

        plt.show()

    def run_quantum_time_series(self, t_series, Mx_series, My_series, Mz_series, M_series):
        if self.integrator != "heun":
            raise ValueError("Quantum noise currently only supports the Heun integrator")

        z5, v5, z6, v6 = init_quantum_noise(self.N)
        evolve_quantum_time_series(self.S, self.nn, self.nnn, z5, v5, z6, v6, t_series, Mx_series, My_series, Mz_series, M_series, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.q_pref, self.dt_q, self.amp5, self.amp6, self.iterations, self.stride)

    def run_classical_time_series(self, t_series, Mx_series, My_series, Mz_series, M_series):
        if self.integrator == "heun":
            evolve_classical_time_series(self.S, self.nn, self.nnn, t_series, Mx_series, My_series, Mz_series, M_series, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.c_pref, self.iterations, self.stride)

        elif self.integrator == "rkmk2":
            evolve_classical_rkmk2_time_series(self.S, self.nn, self.nnn, t_series, Mx_series, My_series, Mz_series, M_series, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.c_pref, self.iterations, self.stride)

        else:
            raise ValueError("integrator must be 'heun' or 'rkmk2'")

    def run_deterministic_time_series(self, t_series, Mx_series, My_series, Mz_series, M_series):
        if self.integrator == "heun":
            evolve_deterministic_heun_time_series(self.S, self.nn, self.nnn, t_series, Mx_series, My_series, Mz_series, M_series, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.iterations, self.stride)

        elif self.integrator == "rkmk2":
            evolve_deterministic_rkmk2_time_series(self.S, self.nn, self.nnn, t_series, Mx_series, My_series, Mz_series, M_series, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.iterations, self.stride)

        else:
            raise ValueError("integrator must be 'heun' or 'rkmk2'")

    def run_spectrum(self):
        t0 = time.time()

        if self.integrator != "heun":
            raise ValueError("Spectrum mode currently only supports the Heun integrator")

        burn_in_steps = int(self.burn_in_time / self.dt)

        if burn_in_steps >= self.iterations:
            raise ValueError("burn_in_time must be smaller than end_time")

        rem = self.iterations - burn_in_steps
        nsamp = (rem + self.stride - 1) // self.stride
        dt_sample = self.stride * self.dt

        Sx_samp = np.empty((nsamp, self.N), dtype=np.float32)
        Sy_samp = np.empty((nsamp, self.N), dtype=np.float32)
        Sz_samp = np.empty((nsamp, self.N), dtype=np.float32)

        if self.noise_mode == "quantum":
            z5, v5, z6, v6 = init_quantum_noise(self.N)
            evolve_quantum_spin_samples(self.S, self.nn, self.nnn, z5, v5, z6, v6, Sx_samp, Sy_samp, Sz_samp, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.q_pref, self.dt_q, self.amp5, self.amp6, self.iterations, burn_in_steps, self.stride)

        elif self.noise_mode == "classical":
            evolve_classical_spin_samples(self.S, self.nn, self.nnn, Sx_samp, Sy_samp, Sz_samp, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.c_pref, self.iterations, burn_in_steps, self.stride)

        elif self.noise_mode == "none":
            evolve_deterministic_spin_samples(self.S, self.nn, self.nnn, Sx_samp, Sy_samp, Sz_samp, self.dt, self.gamma, self.lam, self.J1_mu, self.J2_mu, self.K_mu, self.h_mu, self.iterations, burn_in_steps, self.stride)

        else:
            raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

        if self.spectrum_path not in ("kx", "high_symmetry"):
            raise ValueError("path_mode must be 'kx' or 'high_symmetry'")

        if self.initial_state == "fm":
            plot_fm_magnon_spectrum(Sx_samp, Sy_samp, self.Lx, self.Ly, self.Lz, nsamp, dt_sample, self.J1, self.J2, self.K, self.h, self.gamma, self.mu, path_mode=self.spectrum_path, show_analytic=self.show_analytic)

        elif self.initial_state == "afm":
            plot_afm_magnon_spectrum(Sx_samp, Sy_samp, self.Lx, self.Ly, self.Lz, nsamp, dt_sample, self.J1, self.J2, self.K, self.h, 1.0, self.gamma, self.mu, path_mode=self.spectrum_path, show_analytic=self.show_analytic)

        else:
            raise ValueError("Spectrum mode requires initial_state to be 'fm' or 'afm'")

        t1 = time.time()
        print(f"Simulation time: {t1 - t0:.3f} s")

        plt.show()
