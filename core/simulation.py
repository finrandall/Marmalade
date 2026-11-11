import time
import json

import numpy as np
from numba import get_num_threads

from .constants import (
    BOHR_MAGNETON,
    BOLTZMANN_CONSTANT,
    HBAR,
    MILLIELECTRON_VOLT,
)
from .conventions import HAMILTONIAN_CONVENTION
from .damping import (
    damping_at_time, damping_at_times, damping_noise_factor, classical_noise_factor,
)
from .dipolar import DipolarFFT, evolve_dipolar
from .evolve_classical import evolve_classical_spin_samples, evolve_classical_time_series
from .evolve_quantum import evolve_quantum_spin_samples, evolve_quantum_time_series
from .initialise import INITIAL_STATES, init_spins
from .integrators import INTEGRATORS
from .lattice import SHELL_NAMES, active_bond_graph, build_honeycomb_monolayer
from .thermostats import (
    init_quantum_noise, quantum_noise_amplitudes,
    QUANTUM_THERMOSTAT_INTEGRATOR, QUANTUM_THERMOSTAT_SCHEME,
)
from .observables import compute_energy_shells, energy_of_spin_samples
from .thermostats import initialise_rng_states
from .fft_thermostat import SlidingFFTQuantumThermostat, FFT_THERMOSTAT_SCHEME
from .structure import REFERENCE_LATTICE_CONSTANT_ANGSTROM

OUTPUT_CHOICES = (
    "magnetisation_mean",
    "magnetisation_timeseries",
    "energy_timeseries",
    "spin_trajectory",
    "fourier_modes",
    "spin_snapshots",
)


class Simulation:

    def __init__(self, params, outputs=("magnetisation_mean",)):
        params = dict(params)
        self._time_offset = 0.0
        self._quantum_start = None
        self.lattice_type = str(params.get("lattice", "cubic"))
        if self.lattice_type not in ("cubic", "honeycomb"):
            raise ValueError("lattice must be 'cubic' or 'honeycomb'")
        if self.lattice_type == "cubic":
            params.setdefault("moment_muB", 1.0)
            params.setdefault("dipolar_enabled", False)
            params.setdefault("lattice_constant_angstrom", 1.0)
            params.setdefault("J1_meV", 1000.0 * params.get("J", -0.01))
            if "K" in params:
                raise ValueError("use D in eV for the cubic +D*(s.n)^2 term")
            params.setdefault("D_meV", 1000.0 * params.get("D", 0.0))
            params.setdefault("field_tesla", (
                0.0, 0.0, params.get("h", 0.0) * 1000.0 * MILLIELECTRON_VOLT
                / (params["moment_muB"] * BOHR_MAGNETON),
            ))
            for name in ("J2_meV", "J3_meV", "lambda1_meV",
                         "lambda2_meV", "lambda3_meV"):
                params.setdefault(name, 0.0)
            params["initial_state"] = {
                "fm": "fm_z", "afm": "neel_z",
            }.get(params["initial_state"], params["initial_state"])
        if "hamiltonian_convention" in params:
            raise ValueError("hamiltonian_convention is no longer a simulation parameter")
        self.params = dict(params)
        self.outputs = tuple(dict.fromkeys(outputs))
        unknown_outputs = set(self.outputs) - set(OUTPUT_CHOICES)
        if unknown_outputs:
            raise ValueError(f"unsupported outputs: {sorted(unknown_outputs)}")
        if not self.outputs:
            raise ValueError("at least one output must be selected")

        self.noise_mode = str(params["noise_mode"])
        if self.noise_mode not in ("quantum", "classical", "none", "quantum_fft"):
            raise ValueError("noise_mode must be quantum, quantum_fft, classical or none")
        self.integrator = str(params.get("integrator", "heun"))
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        self.initial_state = str(params["initial_state"])
        if self.initial_state not in INITIAL_STATES:
            raise ValueError(f"initial_state must be one of {INITIAL_STATES}")

        self.T = float(params["T"])
        self.Lx = int(params["Lx"])
        self.Ly = int(params["Ly"])
        self.Lz = int(params.get("Lz", 1))
        self.pbc_x = bool(params["pbc_x"])
        self.pbc_y = bool(params["pbc_y"])
        self.pbc_z = bool(params.get("pbc_z", True))
        if any(name in params for name in
               ("cr_cl_bond_lengths_angstrom", "cr_cl_bond_directions")):
            raise ValueError(
                "Cr-Cl geometry inputs are no longer supported; use "
                "lattice_constant_angstrom for the Cr-only honeycomb"
            )
        self.lattice_constant_angstrom = float(
            params.get("lattice_constant_angstrom",
                       REFERENCE_LATTICE_CONSTANT_ANGSTROM)
        )

        self.dt = float(params["dt"])
        self.end_time = float(params["end_time"])
        self.burn_in_time = float(params["burn_in_time"])
        self.stride = int(params["stride"])
        self.random_seed = params.get("random_seed")
        if self.random_seed is not None:
            self.random_seed = int(self.random_seed)

        self.gamma = float(params.get("gyromagnetic_ratio", 1.76085963023e11))
        self.lam = float(params["lam"])
        initial_damping = params.get("damping_initial")
        self.damping_initial = (
            self.lam if initial_damping is None else float(initial_damping)
        )
        self.damping_decay_time = float(params.get("damping_decay_time", 0.0))
        self.moment_muB = float(params["moment_muB"])
        self.magnetic_moment = self.moment_muB * BOHR_MAGNETON
        self.dipolar_enabled = bool(params.get("dipolar_enabled", True))
        self.dipolar_tolerance = float(params.get("dipolar_tolerance", 1.0e-11))
        self.dipolar_cutoff_angstrom = params.get("dipolar_cutoff_angstrom")
        if self.dipolar_cutoff_angstrom is not None:
            self.dipolar_cutoff_angstrom = float(self.dipolar_cutoff_angstrom)
            if not np.isfinite(self.dipolar_cutoff_angstrom) or self.dipolar_cutoff_angstrom <= 0:
                raise ValueError("dipolar_cutoff_angstrom must be finite and positive, or None")
        self.fft_workers = int(params.get("fft_workers", 1))
        self.J_shell = np.array(
            [params["J1_meV"], params["J2_meV"], params["J3_meV"]],
            dtype=np.float64,
        ) * MILLIELECTRON_VOLT
        self.lambda_shell = np.array(
            [
                params["lambda1_meV"],
                params["lambda2_meV"],
                params["lambda3_meV"],
            ],
            dtype=np.float64,
        ) * MILLIELECTRON_VOLT
        self.J_mu_shell = self.J_shell / self.magnetic_moment
        self.lambda_mu_shell = self.lambda_shell / self.magnetic_moment
        self.D = float(params["D_meV"]) * MILLIELECTRON_VOLT
        self.anisotropy_axis = np.asarray(
            params.get("anisotropy_axis", (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        self.field_tesla = np.asarray(
            params.get("field_tesla", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        self._validate_parameters()

        self.iterations = int(round(self.end_time / self.dt))
        self.burn_in_steps = int(round(self.burn_in_time / self.dt))
        if self.iterations < 1:
            raise ValueError("end_time/dt must produce at least one integration step")
        if self.burn_in_steps >= self.iterations:
            raise ValueError("burn_in_time leaves no integration samples")

        self.D_mu = self.D / self.magnetic_moment
        thermal_energy = BOLTZMANN_CONSTANT * self.T
        self.quantum_noise_scale = np.sqrt(
            2.0 * thermal_energy * thermal_energy
            / (self.gamma * self.magnetic_moment * HBAR)
        )
        self.classical_noise_scale = np.sqrt(
            2.0 * thermal_energy
            / (self.gamma * self.magnetic_moment * self.dt)
        )
        if self.noise_mode == "none":
            self.classical_noise_scale = 0.0
        self.dt_q = self.dt * thermal_energy / HBAR
        noise_factor = damping_noise_factor(self.lam)
        self.q_pref = self.quantum_noise_scale * noise_factor
        self.classical_prefactor = self.classical_noise_scale * classical_noise_factor(self.lam)
        self.amp5, self.amp6 = quantum_noise_amplitudes(self.dt_q)

    def _validate_parameters(self):
        if self.T < 0.0:
            raise ValueError("temperature must be non-negative")
        if min(self.Lx, self.Ly, self.Lz) < 1:
            raise ValueError("Lx, Ly and Lz must all be at least one")
        if self.lattice_type == "honeycomb" and self.Lz != 1:
            raise ValueError("the honeycomb monolayer requires Lz=1")
        if self.lattice_type == "cubic":
            if self.dipolar_enabled:
                raise ValueError("the dipolar FFT currently supports the honeycomb monolayer only")
            if np.any(self.J_shell[1:]) or np.any(self.lambda_shell):
                raise ValueError("the cubic model supports isotropic nearest-neighbour exchange only")
        if (not np.isfinite(self.lattice_constant_angstrom)
                or self.lattice_constant_angstrom <= 0.0):
            raise ValueError("lattice_constant_angstrom must be finite and positive")
        if self.dt <= 0.0 or self.end_time <= 0.0:
            raise ValueError("dt and end_time must be positive")
        if self.burn_in_time < 0.0 or self.burn_in_time >= self.end_time:
            raise ValueError("burn_in_time must be non-negative and below end_time")
        if self.stride < 1:
            raise ValueError("stride must be at least one")
        if self.gamma <= 0.0 or self.moment_muB <= 0.0:
            raise ValueError("gyromagnetic ratio and magnetic moment must be positive")
        if not 0.0 < self.dipolar_tolerance < 1.0:
            raise ValueError("dipolar_tolerance must lie between zero and one")
        if self.lam < 0.0 or self.damping_initial < 0.0:
            raise ValueError("damping values must be non-negative")
        if self.damping_decay_time < 0.0:
            raise ValueError("damping_decay_time must be non-negative")
        if self.anisotropy_axis.shape != (3,):
            raise ValueError("anisotropy_axis must contain three components")
        axis_norm = np.linalg.norm(self.anisotropy_axis)
        if axis_norm == 0.0:
            raise ValueError("anisotropy_axis must be non-zero")
        self.anisotropy_axis /= axis_norm
        if self.field_tesla.shape != (3,):
            raise ValueError("field_tesla must contain three components")

    def initialise(self, include_structure=True):
        get_num_threads()
        if self.lattice_type == "cubic":
            from .lattice import build_simple_cubic
            self.lattice = build_simple_cubic(
                self.Lx, self.Ly, self.Lz,
                lattice_constant_angstrom=self.lattice_constant_angstrom,
                pbc_x=self.pbc_x, pbc_y=self.pbc_y, pbc_z=self.pbc_z,
                include_structure=include_structure,
            )
        else:
            self.lattice = build_honeycomb_monolayer(
                self.Lx,
                self.Ly,
                lattice_constant_angstrom=self.lattice_constant_angstrom,
                pbc_x=self.pbc_x,
                pbc_y=self.pbc_y,
                include_structure=include_structure,
            )
        self.row_ptr, self.neighbours, self.bond_shell = active_bond_graph(
            self.lattice["row_ptr"], self.lattice["neighbours"],
            self.lattice["bond_shell"], self.J_shell, self.lambda_shell,
        )
        self.N = len(self.lattice["basis_indices"])
        self.S = init_spins(
            self.N,
            self.lattice.get("sublattice_indices", self.lattice["basis_indices"]),
            mode=self.initial_state,
            seed=self.random_seed,
        )
        thermostat_seed = (
            None if self.random_seed is None else self.random_seed + 1
        )
        self.thermostat_rng_states = initialise_rng_states(
            self.N, seed=thermostat_seed
        )
        self.dipolar_solver = None
        if self.dipolar_enabled:
            self.dipolar_solver = DipolarFFT(
                self.lattice["lattice_vectors_angstrom"],
                self.lattice["magnetic_basis_angstrom"],
                self.Lx,
                self.Ly,
                self.magnetic_moment,
                pbc_x=self.pbc_x,
                pbc_y=self.pbc_y,
                tolerance=self.dipolar_tolerance,
                workers=self.fft_workers,
                cutoff_angstrom=self.dipolar_cutoff_angstrom,
            )
        self.fft_thermostat = (SlidingFFTQuantumThermostat.from_simulation(self)
                               if self.noise_mode == "quantum_fft" else None)
        self.spatial_recorder = None
        if set(self.outputs) & {"fourier_modes", "spin_snapshots"}:
            from .spatial_observables import SpatialRecorder
            self.spatial_recorder = SpatialRecorder(self)

    def run(self, initial_spins=None, resume=None):
        self.initialise()
        self._time_offset = 0.0
        self._quantum_start = None
        if resume is not None:
            initial_spins = self._apply_checkpoint(resume)
        if initial_spins is not None:
            spins = np.asarray(initial_spins, dtype=np.float64)
            if spins.shape != self.S.shape or not np.all(np.isfinite(spins)):
                raise ValueError("initial_spins must be a finite array with shape (number_of_spins, 3)")
            if not np.allclose(np.linalg.norm(spins, axis=1), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError("initial_spins must contain unit spins")
            self.S[:] = spins
        if "spin_trajectory" in self.outputs:
            data = self.run_spin_trajectory()
            if self.dipolar_solver is None and self.noise_mode != "quantum_fft":
                if "magnetisation_timeseries" in self.outputs or "magnetisation_mean" in self.outputs:
                    self.add_magnetisation_from_trajectory(data)
                if "energy_timeseries" in self.outputs:
                    self.add_energy_from_trajectory(data)
        else:
            data = self.run_observables()
        if self._time_offset and "time" in data:
            data["time"] = data["time"] + self._time_offset
        self.add_damping_metadata(data)
        self.add_structure_metadata(data)
        data.update(self.checkpoint())
        return data


    def checkpoint(self):
        state = {
            "checkpoint_spins": np.array(self.S, dtype=np.float64, copy=True),
            "checkpoint_time": np.array(self._time_offset + self.iterations * self.dt),
            "checkpoint_noise_mode": np.array(self.noise_mode),
            "checkpoint_number_of_spins": np.array(self.N),
            "checkpoint_dt": np.array(self.dt),
            "checkpoint_rng_states": np.array(self.thermostat_rng_states, copy=True),
        }
        if self.noise_mode == "quantum" and getattr(self, "_quantum_arrays", None) is not None:
            for name, array in zip(("z5", "v5", "z6", "v6"), self._quantum_arrays):
                state[f"checkpoint_quantum_{name}"] = np.array(array, copy=True)
        if self.noise_mode == "quantum_fft" and self.fft_thermostat is not None:
            state["checkpoint_fft_state_json"] = np.array(json.dumps(self.fft_thermostat.state_dict()))
        return state

    def _apply_checkpoint(self, checkpoint):
        noise = str(checkpoint["checkpoint_noise_mode"])
        if noise != self.noise_mode:
            raise ValueError(f"checkpoint noise mode {noise!r} differs from this run's {self.noise_mode!r}")
        if int(checkpoint["checkpoint_number_of_spins"]) != self.N:
            raise ValueError("checkpoint has a different number of spins (lattice size)")
        if not np.isclose(float(checkpoint["checkpoint_dt"]), self.dt, rtol=1e-12, atol=0):
            raise ValueError("checkpoint was written with a different timestep")
        self._time_offset = float(checkpoint["checkpoint_time"])
        self.thermostat_rng_states[:] = np.asarray(checkpoint["checkpoint_rng_states"])
        if self.noise_mode == "quantum":
            self._quantum_start = tuple(np.array(checkpoint[f"checkpoint_quantum_{name}"], dtype=np.float64)
                                        for name in ("z5", "v5", "z6", "v6"))
        if self.noise_mode == "quantum_fft":
            self.fft_thermostat.load_state_dict(json.loads(str(checkpoint["checkpoint_fft_state_json"])))
        return np.asarray(checkpoint["checkpoint_spins"], dtype=np.float64)

    def _quantum_initial_state(self):
        state = self._quantum_start
        if state is None:
            state = init_quantum_noise(self.N, seed=self.random_seed)
        self._quantum_arrays = state
        return state

    def add_damping_metadata(self, data):
        data["integrator"] = np.array(self.integrator)
        data["deterministic_integrator_order"] = np.array(4 if self.integrator == "rk4" else 2)
        if self.noise_mode == "classical":
            data["classical_noise_normalisation"] = np.array("gilbert_fluctuation_dissipation")
            data["thermal_field_discretisation"] = np.array("one_gaussian_field_per_step_reused_at_all_stages")
        elif self.noise_mode == "quantum_fft":
            bath = self.fft_thermostat
            data.update({
                "quantum_noise_normalisation": np.array("gilbert_fluctuation_dissipation"),
                "quantum_thermostat_integrator": np.array("offline_fft_overlap_save"),
                "quantum_thermostat_scheme": np.array(FFT_THERMOSTAT_SCHEME),
                "quantum_zero_point_noise": np.array(False),
                "thermal_field_discretisation": np.array("piecewise_linear_coloured_field_at_spin_stages"),
                "quantum_fft_actual_interval_fs": np.array(bath.interval * 1e15),
                "quantum_fft_cache_capacity_bytes": np.array(bath.cache_capacity_bytes),
                "quantum_fft_state_json": np.array(json.dumps(bath.state_dict())),
            })
        elif self.noise_mode == "quantum":
            data["quantum_noise_normalisation"] = np.array("gilbert_fluctuation_dissipation")
            data["quantum_thermostat_integrator"] = np.array(QUANTUM_THERMOSTAT_INTEGRATOR)
            data["quantum_thermostat_scheme"] = np.array(QUANTUM_THERMOSTAT_SCHEME)
        data.update({
            "damping_initial": np.array(self.damping_initial),
            "damping_final": np.array(self.lam),
            "damping_decay_time": np.array(self.damping_decay_time),
        })
        if "time" in data:
            schedule_time = np.maximum(
                np.asarray(data["time"], dtype=np.float64) - self.dt, 0.0
            )
            data["damping_lambda"] = np.where(
                schedule_time < self.burn_in_time,
                damping_at_times(
                    schedule_time, self.damping_initial, self.lam,
                    self.damping_decay_time,
                ),
                self.lam,
            )

    def add_structure_metadata(self, data):
        J_edge = self.J_shell[self.lattice["bond_shell"]]
        lambda_edge = self.lambda_shell[self.lattice["bond_shell"]]
        data.update({
            "model": np.array(
                "cubic_Heisenberg" if self.lattice_type == "cubic" else (
                    "CrCl3_monolayer_exchange_dipolar"
                    if self.dipolar_enabled else "CrCl3_monolayer_exchange"
                )
            ),
            "lattice": np.array(self.lattice_type),
            "lattice_constant_angstrom": np.array(self.lattice_constant_angstrom),
            "lattice_vectors_angstrom": self.lattice["lattice_vectors_angstrom"],
            "supercell_vectors_angstrom": self.lattice["supercell_vectors_angstrom"],
            "basis_fractional": self.lattice["basis_fractional"],
            "magnetic_basis_angstrom": self.lattice["magnetic_basis_angstrom"],
            "positions_angstrom": self.lattice["positions_angstrom"],
            "atomic_positions_angstrom": self.lattice["atomic_positions_angstrom"],
            "atomic_species": self.lattice["atomic_species"],
            "magnetic_atom_indices": self.lattice["magnetic_atom_indices"],
            "atomic_basis_angstrom": self.lattice["atomic_basis_angstrom"],
            "cell_indices": self.lattice["cell_indices"],
            "basis_indices": self.lattice["basis_indices"],
            "bond_row_ptr": self.lattice["row_ptr"],
            "bond_neighbours": self.lattice["neighbours"],
            "bond_shell": self.lattice["bond_shell"],
            "bond_shell_names": np.asarray(SHELL_NAMES),
            "bond_displacement_angstrom": self.lattice["bond_displacement_angstrom"],
            "bond_exchange_meV": J_edge / MILLIELECTRON_VOLT,
            "bond_exchange_anisotropy_meV": (
                lambda_edge / MILLIELECTRON_VOLT
            ),
            "magnetic_moment_muB": np.array(self.moment_muB),
            "magnetisation_units": np.array(
                "muB_per_site" if self.lattice_type == "cubic" else "muB_per_Cr"
            ),
            "spin_units": np.array("unit_vector"),
            "hamiltonian_convention": np.array(HAMILTONIAN_CONVENTION),
            "D_meV": np.array(self.D / MILLIELECTRON_VOLT),
            "anisotropy_axis": self.anisotropy_axis.copy(),
            "field_tesla": self.field_tesla.copy(),
            "dipolar_enabled": np.array(self.dipolar_enabled),
            "dipolar_tolerance": np.array(self.dipolar_tolerance),
        })
        if self.lattice_type == "honeycomb":
            data["nearest_cr_displacements_angstrom"] = self.lattice["nearest_cr_displacements_angstrom"]
        else:
            data["nearest_neighbour_displacements_angstrom"] = self.lattice["nearest_neighbour_displacements_angstrom"]
        if self.dipolar_solver is not None:
            data["dipolar_fft_workers"] = np.array(self.dipolar_solver.fft_workers)
            data["dipolar_fft_backend"] = np.array("scipy")
            for key, value in self.dipolar_solver.metadata.items():
                data[f"dipolar_{key}"] = np.array(value)

    @staticmethod
    def add_magnetisation_summary(data, values):
        count = values.shape[1]
        if count == 0:
            raise ValueError("burn_in_time leaves no magnetisation samples")
        means = values.mean(axis=1)
        errors = values.std(axis=1) / np.sqrt(count)
        data["magnetisation_sample_count"] = np.array(count)
        for component, name in enumerate(("x", "y", "z", "magnitude")):
            data[f"magnetisation_mean_{name}"] = np.array(means[component])
            data[f"magnetisation_standard_error_{name}"] = np.array(errors[component])

    def add_magnetisation_from_trajectory(self, data):
        mx = data["spin_x"].mean(axis=1) * self.moment_muB
        my = data["spin_y"].mean(axis=1) * self.moment_muB
        mz = data["spin_z"].mean(axis=1) * self.moment_muB
        magnitude = np.sqrt(mx * mx + my * my + mz * mz)
        if "magnetisation_timeseries" in self.outputs:
            data.update({
                "magnetisation_x": mx,
                "magnetisation_y": my,
                "magnetisation_z": mz,
                "magnetisation_magnitude": magnitude,
            })
        if "magnetisation_mean" in self.outputs:
            selected = data["time"] >= self.burn_in_time
            self.add_magnetisation_summary(
                data,
                np.vstack((mx[selected], my[selected], mz[selected], magnitude[selected])),
            )

    def add_energy_from_trajectory(self, data):
        if self.dipolar_solver is None:
            data["energy"] = energy_of_spin_samples(
                data["spin_x"], data["spin_y"], data["spin_z"],
                self.row_ptr, self.neighbours, self.bond_shell,
                self.J_shell, self.lambda_shell, self.D,
                self.anisotropy_axis, self.field_tesla, self.magnetic_moment,
            )
            return
        energy = np.empty(len(data["time"]), dtype=np.float64)
        dipolar_energy = np.zeros(len(data["time"]), dtype=np.float64)
        for sample in range(len(energy)):
            spins = np.column_stack((
                data["spin_x"][sample],
                data["spin_y"][sample],
                data["spin_z"][sample],
            ))
            energy[sample] = compute_energy_shells(
                spins, self.row_ptr, self.neighbours, self.bond_shell,
                self.J_shell, self.lambda_shell, self.D,
                self.anisotropy_axis, self.field_tesla, self.magnetic_moment,
            )
            if self.dipolar_solver is not None:
                dipolar_energy[sample] = self.dipolar_solver.energy(spins)
                energy[sample] += dipolar_energy[sample]
        data["energy"] = energy
        if self.dipolar_solver is not None:
            data["dipolar_energy"] = dipolar_energy

    def run_observables(self):
        if self.dipolar_solver is not None or self.noise_mode == "quantum_fft":
            return self._run_observables_dipolar()
        started = time.time()
        requested_magnetisation = "magnetisation_timeseries" in self.outputs
        save_energy = "energy_timeseries" in self.outputs
        save_time_series = requested_magnetisation or save_energy
        burn_in_steps = 0 if save_time_series else self.burn_in_steps
        number_of_samples = (
            self.iterations - burn_in_steps + self.stride - 1
        ) // self.stride
        save_magnetisation = requested_magnetisation or (
            save_time_series and "magnetisation_mean" in self.outputs
        )
        allocation = number_of_samples if save_time_series else 0
        arrays = [np.empty(allocation, dtype=np.float64) for _ in range(6)]
        time_series, mx, my, mz, magnitude, energy = arrays
        moments = np.zeros(9, dtype=np.float64)

        common = (
            self.S, self.row_ptr, self.neighbours, self.bond_shell,
            self.J_mu_shell, self.lambda_mu_shell,
            self.D_mu, self.anisotropy_axis, self.field_tesla,
        )
        tail = (
            self.dt, self.gamma, self.lam, self.J_shell, self.lambda_shell,
            self.D, self.magnetic_moment,
        )
        if self.noise_mode == "quantum":
            z5, v5, z6, v6 = self._quantum_initial_state()
            self.S = evolve_quantum_time_series(
                *common, z5, v5, z6, v6, self.thermostat_rng_states,
                *arrays, moments,
                save_magnetisation, save_energy, *tail, self.q_pref,
                self.quantum_noise_scale, self.damping_initial,
                self.damping_decay_time, self.burn_in_time, self._time_offset,
                self.dt_q, self.amp5, self.amp6, self.iterations, burn_in_steps,
                self.stride, int(self.integrator == "rk4"),
            )
        else:
            self.S = evolve_classical_time_series(
                *common, self.thermostat_rng_states, *arrays, moments,
                save_magnetisation, save_energy, *tail,
                self.classical_prefactor, self.classical_noise_scale,
                self.damping_initial, self.damping_decay_time,
                self.burn_in_time, self._time_offset, self.iterations, burn_in_steps,
                self.stride, int(self.integrator == "rk4"),
            )

        if save_magnetisation:
            mx *= self.moment_muB
            my *= self.moment_muB
            mz *= self.moment_muB
            magnitude *= self.moment_muB

        data = {"elapsed_seconds": np.array(time.time() - started)}
        if save_time_series:
            data["time"] = time_series
        if requested_magnetisation:
            data.update({
                "magnetisation_x": mx,
                "magnetisation_y": my,
                "magnetisation_z": mz,
                "magnetisation_magnitude": magnitude,
            })
        if save_energy:
            data["energy"] = energy
        if "magnetisation_mean" in self.outputs:
            if save_time_series:
                selected = time_series >= self.burn_in_time
                self.add_magnetisation_summary(
                    data,
                    np.vstack((mx[selected], my[selected], mz[selected], magnitude[selected])),
                )
            else:
                count = int(moments[0])
                means = moments[1:5] / count
                variances = np.maximum(moments[5:9] / count - means * means, 0.0)
                means *= self.moment_muB
                variances *= self.moment_muB * self.moment_muB
                data["magnetisation_sample_count"] = np.array(count)
                for component, name in enumerate(("x", "y", "z", "magnitude")):
                    data[f"magnetisation_mean_{name}"] = np.array(means[component])
                    data[f"magnetisation_standard_error_{name}"] = np.array(
                        np.sqrt(variances[component] / count)
                    )
        return data

    def _run_observables_dipolar(self):
        started = time.time()
        requested_magnetisation = "magnetisation_timeseries" in self.outputs
        save_energy = "energy_timeseries" in self.outputs
        save_time_series = requested_magnetisation or save_energy
        burn_in_steps = 0 if save_time_series else self.burn_in_steps
        sample_steps = np.arange(
            burn_in_steps, self.iterations, self.stride, dtype=np.int64
        )
        collect_magnetisation = requested_magnetisation or (
            "magnetisation_mean" in self.outputs
        )
        quantum_state = None
        if self.noise_mode == "quantum":
            quantum_state = self._quantum_initial_state()
        evolved = evolve_dipolar(
            self,
            self.iterations,
            sample_steps,
            quantum_state=quantum_state,
            time_offset=self._time_offset,
            damping_initial=self.damping_initial,
            damping_decay_time=self.damping_decay_time,
            damping_schedule_end_time=self.burn_in_time,
            record_magnetisation=collect_magnetisation,
            record_energy=save_energy,
        )

        data = {"elapsed_seconds": np.array(time.time() - started)}
        if save_time_series:
            data["time"] = evolved["time"]
        self._add_dipolar_observables(data, evolved)
        return data

    def _add_dipolar_observables(self, data, evolved):
        if "magnetisation" in evolved:
            values = evolved["magnetisation"].T * self.moment_muB
            if "magnetisation_timeseries" in self.outputs:
                data.update({
                    "magnetisation_x": values[0],
                    "magnetisation_y": values[1],
                    "magnetisation_z": values[2],
                    "magnetisation_magnitude": values[3],
                })
            if "magnetisation_mean" in self.outputs:
                selected = evolved["time"] >= self.burn_in_time
                self.add_magnetisation_summary(data, values[:, selected])
        if "energy_timeseries" in self.outputs:
            data["energy"] = evolved["energy"]
            data["dipolar_energy"] = evolved["dipolar_energy"]
        if self.spatial_recorder is not None:
            data.update(self.spatial_recorder.arrays())

    def run_spin_trajectory(self):
        if self.dipolar_solver is not None or self.noise_mode == "quantum_fft":
            return self._run_spin_trajectory_dipolar()
        started = time.time()
        save_observables = (
            "magnetisation_timeseries" in self.outputs
            or "energy_timeseries" in self.outputs
        )
        burn_in_steps = 0 if save_observables else self.burn_in_steps
        number_of_samples = (
            self.iterations - burn_in_steps + self.stride - 1
        ) // self.stride
        sx = np.empty((number_of_samples, self.N), dtype=np.float32)
        sy = np.empty((number_of_samples, self.N), dtype=np.float32)
        sz = np.empty((number_of_samples, self.N), dtype=np.float32)
        common = (
            self.S, self.row_ptr, self.neighbours, self.bond_shell,
            self.J_mu_shell, self.lambda_mu_shell,
            self.D_mu, self.anisotropy_axis, self.field_tesla,
        )

        if self.noise_mode == "quantum":
            z5, v5, z6, v6 = self._quantum_initial_state()
            self.S = evolve_quantum_spin_samples(
                *common, z5, v5, z6, v6, self.thermostat_rng_states,
                sx, sy, sz, self.dt,
                self.gamma, self.lam, self.q_pref, self.quantum_noise_scale,
                self.damping_initial, self.damping_decay_time,
                self.burn_in_time, self._time_offset, self.dt_q, self.amp5, self.amp6,
                self.iterations, burn_in_steps, self.stride, int(self.integrator == "rk4"),
            )
        else:
            self.S = evolve_classical_spin_samples(
                *common, self.thermostat_rng_states, sx, sy, sz, self.dt,
                self.gamma, self.lam,
                self.classical_prefactor, self.classical_noise_scale,
                self.damping_initial, self.damping_decay_time,
                self.burn_in_time, self._time_offset, self.iterations, burn_in_steps,
                self.stride, int(self.integrator == "rk4"),
            )

        sample_time = (
            burn_in_steps + np.arange(number_of_samples) * self.stride + 1
        ) * self.dt
        return {
            "time": sample_time,
            "spin_x": sx,
            "spin_y": sy,
            "spin_z": sz,
            "sample_time_step": np.array(self.stride * self.dt),
            "elapsed_seconds": np.array(time.time() - started),
        }

    def _run_spin_trajectory_dipolar(self):
        started = time.time()
        save_observables = (
            "magnetisation_timeseries" in self.outputs
            or "energy_timeseries" in self.outputs
        )
        burn_in_steps = 0 if save_observables else self.burn_in_steps
        sample_steps = np.arange(
            burn_in_steps, self.iterations, self.stride, dtype=np.int64
        )
        quantum_state = None
        if self.noise_mode == "quantum":
            quantum_state = self._quantum_initial_state()
        evolved = evolve_dipolar(
            self,
            self.iterations,
            sample_steps,
            quantum_state=quantum_state,
            time_offset=self._time_offset,
            damping_initial=self.damping_initial,
            damping_decay_time=self.damping_decay_time,
            damping_schedule_end_time=self.burn_in_time,
            record_magnetisation=(
                "magnetisation_timeseries" in self.outputs
                or "magnetisation_mean" in self.outputs
            ),
            record_energy="energy_timeseries" in self.outputs,
            record_spins=True,
        )
        data = {
            "time": evolved["time"],
            "spin_x": evolved["spin_x"],
            "spin_y": evolved["spin_y"],
            "spin_z": evolved["spin_z"],
            "sample_time_step": np.array(self.stride * self.dt),
            "elapsed_seconds": np.array(time.time() - started),
        }
        self._add_dipolar_observables(data, evolved)
        return data
