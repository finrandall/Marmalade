import time

import numpy as np


from .evolve_classical import evolve_classical_time_series
from .evolve_quantum import evolve_quantum_time_series
from .evolve_quantum import evolve_quantum_spin_observable_samples
from .damping import damping_at_time, damping_at_times
from .dipolar import evolve_dipolar
from .thermostats import init_quantum_noise, quantum_thermostat_metadata
from .simulation import Simulation


def _sample_damping(sample_times, simulation, time_offset, damping_initial,
                    damping_decay_time, damping_schedule_end_time):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if damping_schedule_end_time >= 0.0:
        return np.full(sample_times.shape[0], float(simulation.lam))
    return damping_at_times(
        time_offset + np.maximum(sample_times - simulation.dt, 0.0),
        float(damping_initial), float(simulation.lam), float(damping_decay_time),
    )


def advance_thermostat(simulation, noise_state, iterations, stride,
                       time_offset=0.0, damping_schedule=None,
                       record_spins=False):
    sampling_step_offset = int(round(time_offset / simulation.dt))
    if record_spins:
        number_of_samples = (
            (sampling_step_offset + iterations) // stride
            - sampling_step_offset // stride
        )
        if number_of_samples < 1:
            raise ValueError(
                "each equilibration interval must contain a recorded spin frame"
            )
    else:
        number_of_samples = (iterations + stride - 1) // stride

    if simulation.dipolar_solver is not None or simulation.noise_mode == "quantum_fft":
        if record_spins:
            sample_steps = np.asarray([
                step for step in range(iterations)
                if (sampling_step_offset + step + 1) % stride == 0
            ], dtype=np.int64)
        else:
            sample_steps = np.arange(0, iterations, stride, dtype=np.int64)
        if len(sample_steps) != number_of_samples:
            raise RuntimeError("internal dipolar sampling count mismatch")

        if damping_schedule is None:
            damping_initial = simulation.lam
            damping_decay_time = 0.0
            damping_schedule_end_time = 0.0
        else:
            damping_initial = damping_schedule["initial"]
            damping_decay_time = damping_schedule["decay_time"]
            damping_schedule_end_time = -1.0

        evolved = evolve_dipolar(
            simulation,
            iterations,
            sample_steps,
            quantum_state=noise_state,
            time_offset=time_offset,
            damping_initial=damping_initial,
            damping_decay_time=damping_decay_time,
            damping_schedule_end_time=damping_schedule_end_time,
            record_magnetisation=True,
            record_energy=True,
            record_spins=record_spins,
        )
        magnetisation = evolved["magnetisation"] * simulation.moment_muB
        damping_lambda = _sample_damping(
            evolved["time"], simulation, time_offset, damping_initial,
            damping_decay_time, damping_schedule_end_time,
        )
        data = {
            "time": evolved["time"],
            "magnetisation_x": magnetisation[:, 0],
            "magnetisation_y": magnetisation[:, 1],
            "magnetisation_z": magnetisation[:, 2],
            "magnetisation_magnitude": magnetisation[:, 3],
            "energy": evolved["energy"],
            "dipolar_energy": evolved["dipolar_energy"],
            "damping_lambda": damping_lambda,
        }
        if record_spins:
            data.update({
                "spin_x": evolved["spin_x"],
                "spin_y": evolved["spin_y"],
                "spin_z": evolved["spin_z"],
            })
        if simulation.noise_mode == "quantum_fft":
            simulation.add_damping_metadata(data)
            data["damping_lambda"] = damping_lambda
        elif simulation.noise_mode == "quantum":
            data.update(quantum_thermostat_metadata())
        return data

    arrays = [np.empty(number_of_samples, dtype=np.float64) for _ in range(6)]
    moments = np.zeros(9, dtype=np.float64)
    common = (
        simulation.S,
        simulation.row_ptr,
        simulation.neighbours,
        simulation.bond_shell,
        simulation.J_mu_shell,
        simulation.lambda_mu_shell,
        simulation.D_mu,
        simulation.anisotropy_axis,
        simulation.field_tesla,
    )
    tail = (
        simulation.dt,
        simulation.gamma,
        simulation.lam,
        simulation.J_shell,
        simulation.lambda_shell,
        simulation.D,
        simulation.magnetic_moment,
    )
    if damping_schedule is None:
        damping_initial = simulation.lam
        damping_decay_time = 0.0
        damping_schedule_end_time = 0.0
    else:
        damping_initial = damping_schedule["initial"]
        damping_decay_time = damping_schedule["decay_time"]
        damping_schedule_end_time = -1.0

    spin_arrays = None
    if record_spins:
        spin_arrays = [
            np.empty((number_of_samples, simulation.N), dtype=np.float32)
            for _ in range(3)
        ]

    if simulation.noise_mode == "quantum":
        z5, v5, z6, v6 = noise_state
        if record_spins:
            simulation.S = evolve_quantum_spin_observable_samples(
                *common,
                z5,
                v5,
                z6,
                v6,
                simulation.thermostat_rng_states,
                *arrays,
                moments,
                *spin_arrays,
                *tail,
                simulation.q_pref,
                simulation.quantum_noise_scale,
                damping_initial,
                damping_decay_time,
                damping_schedule_end_time,
                time_offset,
                simulation.dt_q,
                simulation.amp5,
                simulation.amp6,
                iterations,
                sampling_step_offset,
                stride,
                int(simulation.integrator == "rk4"),
            )
        else:
            simulation.S = evolve_quantum_time_series(
                *common,
                z5,
                v5,
                z6,
                v6,
                simulation.thermostat_rng_states,
                *arrays,
                moments,
                True,
                True,
                *tail,
                simulation.q_pref,
                simulation.quantum_noise_scale,
                damping_initial,
                damping_decay_time,
                damping_schedule_end_time,
                time_offset,
                simulation.dt_q,
                simulation.amp5,
                simulation.amp6,
                iterations,
                0,
                stride,
                int(simulation.integrator == "rk4"),
            )
    elif simulation.noise_mode == "classical":
        if record_spins:
            raise ValueError(
                "equilibration spin recording currently requires quantum noise"
            )
        simulation.S = evolve_classical_time_series(
            *common,
            simulation.thermostat_rng_states,
            *arrays,
            moments,
            True,
            True,
            *tail,
            simulation.classical_prefactor,
            simulation.classical_noise_scale,
            damping_initial,
            damping_decay_time,
            damping_schedule_end_time,
            time_offset,
            iterations,
            0,
            stride,
            int(simulation.integrator == "rk4"),
        )
    else:
        raise ValueError("thermostat must be 'quantum' or 'classical'")

    time_series, mx, my, mz, magnetisation, energy = arrays
    mx *= simulation.moment_muB
    my *= simulation.moment_muB
    mz *= simulation.moment_muB
    magnetisation *= simulation.moment_muB
    damping_lambda = _sample_damping(
        time_series, simulation, time_offset, damping_initial,
        damping_decay_time, damping_schedule_end_time,
    )
    data = {
        "time": time_series,
        "magnetisation_x": mx,
        "magnetisation_y": my,
        "magnetisation_z": mz,
        "magnetisation_magnitude": magnetisation,
        "energy": energy,
        "damping_lambda": damping_lambda,
    }
    if record_spins:
        data.update({
            "spin_x": spin_arrays[0],
            "spin_y": spin_arrays[1],
            "spin_z": spin_arrays[2],
        })
    if simulation.noise_mode == "quantum":
        data.update(quantum_thermostat_metadata())
    return data


def initialise_ordered(params):
    simulation = Simulation(
        params, outputs=("magnetisation_timeseries", "energy_timeseries")
    )
    simulation.initialise(include_structure=False)
    noise_state = None
    if simulation.noise_mode == "quantum":
        noise_state = init_quantum_noise(simulation.N, seed=simulation.random_seed)
    return simulation, noise_state


def equilibrate_ordered(params, checker, record_spin_snapshots=False):
    simulation, noise_state = initialise_ordered(params)
    dt = simulation.dt
    minimum_steps = int(round(checker["min_time"] / dt))
    maximum_steps = int(round(checker["max_time"] / dt))
    check_steps = int(round(checker["check_interval"] / dt))
    window = int(checker["window"])
    consecutive_required = int(checker["consecutive"])
    damping_schedule = {
        "initial": checker["damping_initial"],
        "decay_time": checker["damping_decay_time"],
    }
    energy_scale = simulation.N * abs(simulation.J_shell[0])

    check_time = []
    check_magnetisation = []
    check_magnetisation_reduced = []
    check_energy = []
    magnetisation_change = []
    energy_change = []
    series = []
    initial_spin_snapshot = None
    if record_spin_snapshots:
        initial_spin_snapshot = simulation.S.copy()
    completed_steps = 0
    consecutive = 0
    converged = False

    while completed_steps < maximum_steps:
        chunk_steps = min(check_steps, maximum_steps - completed_steps)
        sample = advance_thermostat(
            simulation, noise_state, chunk_steps, params["stride"],
            completed_steps * dt, damping_schedule,
            record_spins=record_spin_snapshots,
        )
        sample["time"] += completed_steps * dt
        series.append(sample)
        completed_steps += chunk_steps
        interval_magnetisation = float(np.mean(sample["magnetisation_magnitude"]))
        reduced_magnetisation = (
            interval_magnetisation / simulation.moment_muB
        )
        interval_energy = float(np.mean(sample["energy"])) / energy_scale
        check_time.append(completed_steps * dt)
        check_magnetisation.append(interval_magnetisation)
        check_magnetisation_reduced.append(reduced_magnetisation)
        check_energy.append(interval_energy)

        if len(check_time) < 2 * window or len(check_time) % window != 0:
            magnetisation_change.append(np.nan)
            energy_change.append(np.nan)
            continue

        previous = slice(-2 * window, -window)
        recent = slice(-window, None)
        magnetisation_delta = abs(
            np.mean(check_magnetisation_reduced[recent])
            - np.mean(check_magnetisation_reduced[previous])
        )
        energy_delta = abs(
            np.mean(check_energy[recent]) - np.mean(check_energy[previous])
        )
        magnetisation_change.append(magnetisation_delta)
        energy_change.append(energy_delta)

        if completed_steps < minimum_steps:
            continue
        if (
            magnetisation_delta <= checker["magnetisation_tolerance"]
            and energy_delta <= checker["energy_tolerance"]
        ):
            consecutive += 1
            if consecutive >= consecutive_required:
                converged = True
                break
        else:
            consecutive = 0

    equilibration = {
        key: (np.concatenate([sample[key] for sample in series])
              if np.asarray(series[0][key]).ndim else series[-1][key])
        for key in series[0]
    }
    diagnostics = {
        "equilibration_time": np.array(completed_steps * dt),
        "equilibration_converged": np.array(converged),
        "equilibration_check_time": np.asarray(check_time),
        "equilibration_check_magnetisation": np.asarray(check_magnetisation),
        "equilibration_check_magnetisation_reduced": np.asarray(
            check_magnetisation_reduced
        ),
        "equilibration_check_energy": np.asarray(check_energy),
        "equilibration_magnetisation_change": np.asarray(
            magnetisation_change
        ),
        "equilibration_energy_change": np.asarray(energy_change),
        "equilibration_check_statistic": np.array("interval_mean"),
        "equilibration_window_duration": np.array(
            window * checker["check_interval"]
        ),
        "equilibration_test_interval": np.array(
            window * checker["check_interval"]
        ),
        "equilibration_consecutive_achieved": np.array(consecutive),
        "equilibration_time_series": equilibration["time"],
        "equilibration_magnetisation_x": equilibration["magnetisation_x"],
        "equilibration_magnetisation_y": equilibration["magnetisation_y"],
        "equilibration_magnetisation_z": equilibration["magnetisation_z"],
        "equilibration_magnetisation_magnitude": equilibration[
            "magnetisation_magnitude"
        ],
        "equilibration_energy": equilibration["energy"],
        "equilibration_damping_lambda": equilibration["damping_lambda"],
        "damping_initial": np.array(checker["damping_initial"]),
        "damping_final": np.array(simulation.lam),
        "damping_decay_time": np.array(checker["damping_decay_time"]),
        "energy_scale": np.array(energy_scale),
        "number_of_spins": np.array(simulation.N),
        "magnetic_moment_muB": np.array(simulation.moment_muB),
        "magnetisation_units": np.array("muB_per_Cr"),
        "energy_units": np.array("joule_total"),
    }
    if record_spin_snapshots:
        frame_time_parts = [
            np.zeros(1, dtype=np.float64), equilibration["time"],
        ]
        spin_x_parts = [initial_spin_snapshot[None, :, 0],
                        equilibration["spin_x"]]
        spin_y_parts = [initial_spin_snapshot[None, :, 1],
                        equilibration["spin_y"]]
        spin_z_parts = [initial_spin_snapshot[None, :, 2],
                        equilibration["spin_z"]]
        if completed_steps % params["stride"] != 0:
            frame_time_parts.append(np.asarray((completed_steps * dt,)))
            spin_x_parts.append(simulation.S[None, :, 0])
            spin_y_parts.append(simulation.S[None, :, 1])
            spin_z_parts.append(simulation.S[None, :, 2])
        diagnostics.update({
            "equilibration_frame_time": np.concatenate(frame_time_parts),
            "equilibration_spin_x": np.vstack(spin_x_parts).astype(np.float32),
            "equilibration_spin_y": np.vstack(spin_y_parts).astype(np.float32),
            "equilibration_spin_z": np.vstack(spin_z_parts).astype(np.float32),
        })
    return simulation, noise_state, diagnostics


def run_adaptive_thermostat(params, checker, production_time):
    started = time.time()
    simulation, noise_state, diagnostics = equilibrate_ordered(params, checker)
    production_steps = int(round(production_time / params["dt"]))
    data = advance_thermostat(
        simulation, noise_state, production_steps, params["stride"],
        time_offset=(simulation.fft_thermostat.fine_step * simulation.dt
                     if simulation.noise_mode == "quantum_fft" else 0.0),
    )
    data["elapsed_seconds"] = np.array(time.time() - started)
    data.update(diagnostics)
    return data
