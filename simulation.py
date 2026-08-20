import time

import numpy as np

from evolve import evolve_classical_spin_samples
from evolve import evolve_classical_time_series, evolve_deterministic_heun_time_series
from evolve import evolve_deterministic_spin_samples
from evolve import evolve_quantum_spin_samples, evolve_quantum_time_series
from initialise import init_spins
from lattice import build_neighbour_list
from noise import init_quantum_noise, quantum_noise_amplitudes
from observables import compute_energy


class Simulation:
    def __init__(self, params, outputs=("magnetisation_mean",)):
        self.params = params
        self.outputs = tuple(outputs)
        self.noise_mode = params["noise_mode"]
        self.integrator = params["integrator"]
        if self.integrator != "heun":
            raise ValueError("ASD currently supports only the Heun integrator")
        self.initial_state = params["initial_state"]
        self.T = params["T"]
        self.Lx, self.Ly, self.Lz = params["Lx"], params["Ly"], params["Lz"]
        self.pbc_x, self.pbc_y, self.pbc_z = params["pbc_x"], params["pbc_y"], params["pbc_z"]
        self.N = self.Lx * self.Ly * self.Lz
        self.dt = params["dt"]
        self.end_time = params["end_time"]
        self.burn_in_time = params["burn_in_time"]
        self.iterations = int(round(self.end_time / self.dt))
        self.stride = params["stride"]

        self.mu = 9.274e-24
        self.gamma = 1.7609e11
        self.lam = params["lam"]
        self.hbar = 1.054571817e-34
        self.kB = 1.380649e-23
        self.eV = 1.602176634e-19
        self.J = params["J"] * self.eV
        self.K = params["K"] * self.eV
        self.h = params["h"] * self.eV
        self.J_mu = self.J / self.mu
        self.K_mu, self.h_mu = self.K / self.mu, self.h / self.mu
        self.kT = self.kB * self.T
        self.Gamma = self.lam * self.kT / ((1.0 + self.lam * self.lam) * self.gamma * self.mu)
        self.dt_q = self.dt * self.kT / self.hbar
        self.q_pref = np.sqrt(2.0 * self.Gamma * self.kT / self.hbar)
        self.c_pref = np.sqrt(2.0 * self.Gamma / self.dt)
        self.amp5, self.amp6 = quantum_noise_amplitudes(self.dt_q)

    def initialise(self):
        self.S = init_spins(self.Lx, self.Ly, self.Lz, mode=self.initial_state)
        self.nn = build_neighbour_list(
            self.Lx, self.Ly, self.Lz, self.pbc_x, self.pbc_y, self.pbc_z)

    def run(self):
        self.initialise()
        if "spin_trajectory" in self.outputs:
            data = self.run_spin_trajectory()
            if "magnetisation_timeseries" in self.outputs or "magnetisation_mean" in self.outputs:
                self.add_magnetisation_from_trajectory(data)
            if "energy_timeseries" in self.outputs:
                self.add_energy_from_trajectory(data)
            return data
        return self.run_magnetisation()

    def add_magnetisation_from_trajectory(self, data):
        mx = data["spin_x"].mean(axis=1)
        my = data["spin_y"].mean(axis=1)
        mz = data["spin_z"].mean(axis=1)
        magnitude = np.sqrt(mx * mx + my * my + mz * mz)
        if "magnetisation_timeseries" in self.outputs:
            data.update({
                "magnetisation_x": mx, "magnetisation_y": my, "magnetisation_z": mz,
                "magnetisation_magnitude": magnitude,
            })
        if "magnetisation_mean" in self.outputs:
            selected = data["time"] >= self.burn_in_time
            self.add_magnetisation_summary(
                data, np.vstack((mx[selected], my[selected], mz[selected], magnitude[selected])))

    def add_energy_from_trajectory(self, data):
        energy = np.empty(len(data["time"]), dtype=np.float64)
        for sample in range(len(energy)):
            spins = np.column_stack((
                data["spin_x"][sample], data["spin_y"][sample], data["spin_z"][sample]))
            energy[sample] = compute_energy(spins, self.nn, self.J, self.K, self.h)
        data["energy"] = energy

    @staticmethod
    def add_magnetisation_summary(data, values):
        count = values.shape[1]
        if count == 0:
            raise ValueError("burn_in_time leaves no magnetisation samples")
        means = values.mean(axis=1)
        errors = values.std(axis=1) / np.sqrt(count)
        names = ("x", "y", "z", "magnitude")
        data["magnetisation_sample_count"] = np.array(count)
        for i, name in enumerate(names):
            data[f"magnetisation_mean_{name}"] = np.array(means[i])
            data[f"magnetisation_standard_error_{name}"] = np.array(errors[i])

    def run_magnetisation(self):
        started = time.time()
        requested_magnetisation = "magnetisation_timeseries" in self.outputs
        save_energy = "energy_timeseries" in self.outputs
        save_time_series = requested_magnetisation or save_energy
        burn_in_steps = 0 if save_time_series else int(round(self.burn_in_time / self.dt))
        n_samples = (self.iterations - burn_in_steps + self.stride - 1) // self.stride
        save_magnetisation = requested_magnetisation or (
            save_time_series and "magnetisation_mean" in self.outputs)
        allocation = n_samples if save_time_series else 0
        arrays = [np.empty(allocation, dtype=np.float64) for _ in range(6)]
        t_series, mx, my, mz, magnitude, energy = arrays
        moments = np.zeros(9, dtype=np.float64)
        if self.noise_mode == "quantum":
            if self.integrator != "heun":
                raise ValueError("Quantum noise currently only supports the Heun integrator")
            z5, v5, z6, v6 = init_quantum_noise(self.N)
            self.S = evolve_quantum_time_series(self.S, self.nn, z5, v5, z6, v6, *arrays, moments,
                save_magnetisation, save_energy,
                self.dt, self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu,
                self.J, self.K, self.h,
                self.q_pref, self.dt_q, self.amp5, self.amp6, self.iterations, burn_in_steps, self.stride)
        elif self.noise_mode == "classical":
            self.S = evolve_classical_time_series(self.S, self.nn, *arrays, moments, save_magnetisation, save_energy,
                self.dt, self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu, self.J, self.K, self.h,
                self.c_pref, self.iterations, burn_in_steps, self.stride)
        elif self.noise_mode == "none":
            self.S = evolve_deterministic_heun_time_series(self.S, self.nn, *arrays, moments, save_magnetisation, save_energy,
                self.dt, self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu, self.J, self.K, self.h,
                self.iterations, burn_in_steps, self.stride)
        else:
            raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

        data = {"elapsed_seconds": np.array(time.time() - started)}
        if save_time_series:
            data["time"] = t_series
        if requested_magnetisation:
            data.update({
                "magnetisation_x": mx, "magnetisation_y": my,
                "magnetisation_z": mz, "magnetisation_magnitude": magnitude,
            })
        if save_energy:
            data["energy"] = energy
        if "magnetisation_mean" in self.outputs:
            if save_time_series:
                selected = t_series >= self.burn_in_time
                self.add_magnetisation_summary(
                    data, np.vstack((mx[selected], my[selected], mz[selected], magnitude[selected])))
            else:
                count = int(moments[0])
                means = moments[1:5] / count
                variances = np.maximum(moments[5:9] / count - means * means, 0.0)
                data["magnetisation_sample_count"] = np.array(count)
                for i, name in enumerate(("x", "y", "z", "magnitude")):
                    data[f"magnetisation_mean_{name}"] = np.array(means[i])
                    data[f"magnetisation_standard_error_{name}"] = np.array(np.sqrt(variances[i] / count))
        return data

    def run_spin_trajectory(self):
        started = time.time()
        if self.integrator != "heun":
            raise ValueError("Spin trajectory output currently only supports the Heun integrator")
        save_observables = (
            "magnetisation_timeseries" in self.outputs or "energy_timeseries" in self.outputs)
        burn_in_steps = 0 if save_observables else int(round(self.burn_in_time / self.dt))
        remaining = self.iterations - burn_in_steps
        n_samples = (remaining + self.stride - 1) // self.stride
        sx = np.empty((n_samples, self.N), dtype=np.float32)
        sy = np.empty((n_samples, self.N), dtype=np.float32)
        sz = np.empty((n_samples, self.N), dtype=np.float32)
        if self.noise_mode == "quantum":
            z5, v5, z6, v6 = init_quantum_noise(self.N)
            evolve_quantum_spin_samples(self.S, self.nn, z5, v5, z6, v6, sx, sy, sz,
                self.dt, self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu,
                self.q_pref, self.dt_q, self.amp5, self.amp6, self.iterations, burn_in_steps, self.stride)
        elif self.noise_mode == "classical":
            evolve_classical_spin_samples(self.S, self.nn, sx, sy, sz, self.dt,
                self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu,
                self.c_pref, self.iterations, burn_in_steps, self.stride)
        elif self.noise_mode == "none":
            evolve_deterministic_spin_samples(self.S, self.nn, sx, sy, sz, self.dt,
                self.gamma, self.lam, self.J_mu, self.K_mu, self.h_mu,
                self.iterations, burn_in_steps, self.stride)
        else:
            raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")
        sample_time = (burn_in_steps + np.arange(n_samples) * self.stride + 1) * self.dt
        return {
            "time": sample_time, "spin_x": sx, "spin_y": sy, "spin_z": sz,
            "sample_time_step": np.array(self.stride * self.dt),
            "elapsed_seconds": np.array(time.time() - started),
        }
