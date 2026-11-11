import hashlib
import json

import numpy as np
from scipy.fft import fft2

SPATIAL_OUTPUTS = ("fourier_modes", "spin_snapshots")


def requested_spatial_outputs(params):
    return tuple(name for name, flag in (
        ("fourier_modes", "record_fourier_modes"),
        ("spin_snapshots", "record_spin_snapshots"),
    ) if params.get(flag, False))


def interval_steps(interval, dt, name):
    interval = float(interval)
    ratio = interval / dt
    if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-8):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return int(round(ratio))


def low_wavevector_indices(radius, Lx, Ly):
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("fourier_mode_radius must be a non-negative integer")
    radius = int(radius)
    return np.array([
        (h, k)
        for h in range(-min(radius, Lx // 2), min(radius, Lx // 2) + 1)
        for k in range(-min(radius, Ly // 2), min(radius, Ly // 2) + 1)
        if max(abs(h), abs(k), abs(h + k)) <= radius
        and (h > 0 or (h == 0 and k >= 0))
    ], dtype=np.int64).reshape(-1, 2)


def snapshot_positions(data):
    Lx, Ly = map(int, data["spatial_lattice_shape"])
    vectors = np.asarray(data["spatial_lattice_vectors_angstrom"])
    basis = np.asarray(data["spatial_magnetic_basis_angstrom"])
    iy, ix = np.indices((Ly, Lx))
    cells = ix[..., None] * vectors[0] + iy[..., None] * vectors[1]
    return (cells[..., None, :] + basis).reshape(-1, 3)


class SpatialRecorder:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fourier_enabled = "fourier_modes" in simulation.outputs
        self.snapshots_enabled = "spin_snapshots" in simulation.outputs
        if simulation.lattice_type != "honeycomb" or (
            simulation.dipolar_solver is None and simulation.noise_mode != "quantum_fft"
        ):
            raise ValueError("spatial outputs currently require the dipolar honeycomb integrator")
        if self.fourier_enabled and not (simulation.pbc_x and simulation.pbc_y):
            raise ValueError("Fourier modes currently require periodic x and y boundaries")
        self.vectors = simulation.lattice["lattice_vectors_angstrom"].copy()
        self.basis = simulation.lattice["magnetic_basis_angstrom"].copy()
        self.dt = simulation.dt
        self.total_steps = simulation.iterations
        self.fourier_steps = np.empty(0, dtype=np.int64)
        self.snapshot_steps = np.empty(0, dtype=np.int64)
        self.mode_indices = np.empty((0, 2), dtype=np.int64)
        self.wavevectors = np.empty((0, 3))
        self.fourier_stride = 0
        self.snapshot_stride = 0
        if self.fourier_enabled:
            self.fourier_stride = interval_steps(
                simulation.params.get("fourier_interval", 1e-12), self.dt, "fourier_interval"
            )
            explicit = simulation.params.get("fourier_mode_indices")
            if explicit is None:
                indices = low_wavevector_indices(
                    simulation.params.get("fourier_mode_radius", 6), simulation.Lx, simulation.Ly
                )
            else:
                indices = np.asarray(explicit)
                if (indices.ndim != 2 or indices.shape[1] != 2 or len(indices) == 0
                        or not np.all(np.isfinite(indices)) or np.any(indices != np.rint(indices))):
                    raise ValueError("fourier_mode_indices must be a non-empty array of integer (h,k) pairs")
                indices = indices.astype(np.int64)
                if len(np.unique(indices, axis=0)) != len(indices):
                    raise ValueError("fourier_mode_indices contains duplicate wavevectors")
            reciprocal = 2 * np.pi * np.linalg.inv(self.vectors[:2, :2]).T
            qxy = (indices / np.array([simulation.Lx, simulation.Ly])) @ reciprocal
            order = np.argsort(np.sum(qxy**2, axis=1), kind="stable")
            self.mode_indices = indices[order]
            self.wavevectors = np.column_stack((qxy[order], np.zeros(len(indices))))
            self.basis_phases = np.exp(-1j * self.wavevectors @ self.basis.T)
            self.fourier_steps = np.arange(0, self.total_steps + 1, self.fourier_stride, dtype=np.int64)
        if self.snapshots_enabled:
            self.snapshot_stride = interval_steps(
                simulation.params.get("snapshot_interval", .5e-9), self.dt, "snapshot_interval"
            )
            self.snapshot_steps = np.unique(np.r_[
                np.arange(0, self.total_steps + 1, self.snapshot_stride, dtype=np.int64),
                self.total_steps,
            ])
        self.modes = np.empty((len(self.fourier_steps), len(self.mode_indices), 3), dtype=np.complex128)
        self.snapshots = np.empty((len(self.snapshot_steps), simulation.N, 3), dtype=np.float32)
        self.fourier_count = 0
        self.snapshot_count = 0
        signature = dict(
            version=1, shape=[simulation.Lx, simulation.Ly], N=simulation.N,
            dt=self.dt, total_steps=self.total_steps, moment_muB=simulation.moment_muB,
            vectors=self.vectors.tolist(), basis=self.basis.tolist(),
            modes=self.mode_indices.tolist(), fourier_stride=self.fourier_stride,
            snapshot_stride=self.snapshot_stride,
        )
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.next_step = self._next_step()

    def _next_step(self):
        steps = []
        if self.fourier_count < len(self.fourier_steps):
            steps.append(self.fourier_steps[self.fourier_count])
        if self.snapshot_count < len(self.snapshot_steps):
            steps.append(self.snapshot_steps[self.snapshot_count])
        return min(steps) if steps else None

    def record(self, step, spins):
        if self.next_step is not None and step > self.next_step:
            raise ValueError("spatial recording skipped a scheduled step; check restored history")
        if step != self.next_step:
            return
        if self.fourier_count < len(self.fourier_steps) and step == self.fourier_steps[self.fourier_count]:
            sim = self.simulation
            cell_fft = fft2(spins.reshape(sim.Ly, sim.Lx, 2, 3), axes=(0, 1), workers=sim.fft_workers)
            h, k = self.mode_indices.T
            selected = cell_fft[k % sim.Ly, h % sim.Lx]
            self.modes[self.fourier_count] = (
                np.sum(selected * self.basis_phases[..., None], axis=1) * (sim.moment_muB / sim.N)
            )
            self.fourier_count += 1
        if self.snapshot_count < len(self.snapshot_steps) and step == self.snapshot_steps[self.snapshot_count]:
            self.snapshots[self.snapshot_count] = spins
            self.snapshot_count += 1
        self.next_step = self._next_step()

    def arrays(self):
        data = dict(
            spatial_schema_version=np.array(1),
            spatial_recording_signature=np.array(self.signature),
            spatial_lattice_shape=np.array([self.simulation.Lx, self.simulation.Ly]),
            spatial_number_of_spins=np.array(self.simulation.N),
            spatial_moment_muB=np.array(self.simulation.moment_muB),
            spatial_lattice_vectors_angstrom=self.vectors,
            spatial_magnetic_basis_angstrom=self.basis,
            spatial_site_order=np.array("basis + 2*(ix + Lx*iy)"),
        )
        if self.fourier_enabled:
            data.update(
                fourier_time=self.fourier_steps[:self.fourier_count] * self.dt,
                fourier_magnetisation=self.modes[:self.fourier_count],
                fourier_mode_indices=self.mode_indices,
                fourier_wavevectors_inv_angstrom=self.wavevectors,
                fourier_components=np.array(["x", "y", "z"]),
                fourier_units=np.array("muB_per_Cr"),
                fourier_normalisation=np.array("moment_muB/N"),
                fourier_phase_convention=np.array("exp(-1j*q.r)"),
                fourier_sample_interval=np.array(self.fourier_stride * self.dt),
            )
        if self.snapshots_enabled:
            data.update(
                snapshot_time=self.snapshot_steps[:self.snapshot_count] * self.dt,
                snapshot_spins=self.snapshots[:self.snapshot_count],
                snapshot_units=np.array("dimensionless_unit_vector"),
                snapshot_components=np.array(["x", "y", "z"]),
                snapshot_interval=np.array(self.snapshot_stride * self.dt),
            )
        return data

    def restore(self, archive, completed_steps):
        if "spatial_recording_signature" not in archive:
            if completed_steps == 0:
                return
            raise ValueError("checkpoint has no spatial history; start a fresh campaign to record it")
        if str(archive["spatial_recording_signature"]) != self.signature:
            raise ValueError("checkpoint spatial recording settings do not match")
        restored = []
        for enabled, schedule, time_key, values_key, destination in (
            (self.fourier_enabled, self.fourier_steps, "fourier_time", "fourier_magnetisation", self.modes),
            (self.snapshots_enabled, self.snapshot_steps, "snapshot_time", "snapshot_spins", self.snapshots),
        ):
            count = int(np.searchsorted(schedule, completed_steps, side="right")) if enabled else 0
            if enabled:
                times = archive[time_key]
                values = archive[values_key]
                if (times.shape != (count,) or values.shape != destination[:count].shape
                        or not np.array_equal(times, schedule[:count] * self.dt)
                        or not np.all(np.isfinite(values))):
                    raise ValueError(f"invalid checkpoint spatial history: {values_key}")
                destination[:count] = values
            restored.append(count)
        self.fourier_count, self.snapshot_count = restored
        self.next_step = self._next_step()
