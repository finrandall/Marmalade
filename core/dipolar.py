import math
import operator

import numpy as np
from numba import get_num_threads, njit, prange
from scipy import fft as scipy_fft

from .constants import ANGSTROM, VACUUM_PERMEABILITY


_SQRT_PI = math.sqrt(math.pi)
_KERNEL_CACHE = {}


@njit(cache=True, inline="always")
def _planar_spectral_site(kernel, spins, field, index):
    a, b, c = spins[index, 0], spins[index, 1], spins[index, 2]
    d, e, f = spins[index, 3], spins[index, 4], spins[index, 5]
    field[index, 0] = kernel[index, 0]*a + kernel[index, 1]*b + kernel[index, 2]*d + kernel[index, 3]*e
    field[index, 1] = kernel[index, 4]*a + kernel[index, 5]*b + kernel[index, 6]*d + kernel[index, 7]*e
    field[index, 2] = kernel[index, 8]*c + kernel[index, 9]*f
    field[index, 3] = kernel[index, 10]*a + kernel[index, 11]*b + kernel[index, 12]*d + kernel[index, 13]*e
    field[index, 4] = kernel[index, 14]*a + kernel[index, 15]*b + kernel[index, 16]*d + kernel[index, 17]*e
    field[index, 5] = kernel[index, 18]*c + kernel[index, 19]*f


@njit(cache=True)
def _contract_planar_serial(kernel, spins, field):
    packed = kernel.reshape(-1, 20)
    transformed = spins.reshape(-1, 6)
    result = field.reshape(-1, 6)
    for index in range(transformed.shape[0]):
        _planar_spectral_site(packed, transformed, result, index)


@njit(cache=True, parallel=True)
def _contract_planar_parallel(kernel, spins, field):
    packed = kernel.reshape(-1, 20)
    transformed = spins.reshape(-1, 6)
    result = field.reshape(-1, 6)
    for index in prange(transformed.shape[0]):
        _planar_spectral_site(packed, transformed, result, index)


def _validate_planar_geometry(lattice_vectors, magnetic_basis):
    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    basis = np.asarray(magnetic_basis, dtype=np.float64)
    if vectors.shape != (2, 3):
        raise ValueError("lattice_vectors must have shape (2, 3)")
    if basis.shape != (2, 3):
        raise ValueError("magnetic_basis must have shape (2, 3)")
    if not np.allclose(vectors[:, 2], 0.0, atol=1.0e-12):
        raise ValueError("the 2D Ewald solver requires in-plane lattice vectors")
    if not np.allclose(basis[:, 2], basis[0, 2], atol=1.0e-12):
        raise ValueError("the 2D Ewald solver requires coplanar magnetic sites")
    area = abs(np.linalg.det(vectors[:, :2]))
    if area <= 0.0:
        raise ValueError("the in-plane lattice vectors must be independent")
    return np.ascontiguousarray(vectors), np.ascontiguousarray(basis)


def _base_displacements(lattice_vectors, magnetic_basis, Lx, Ly):
    positions = np.empty((Ly, Lx, 2, 2, 2), dtype=np.float64)
    a1 = lattice_vectors[0, :2]
    a2 = lattice_vectors[1, :2]
    tau = magnetic_basis[:, :2]
    for dy in range(Ly):
        for dx in range(Lx):
            cell = dx * a1 + dy * a2
            for target_basis in range(2):
                for source_basis in range(2):
                    positions[dy, dx, target_basis, source_basis] = (
                        cell + tau[target_basis] - tau[source_basis]
                    )
    return positions


@njit(parallel=True, cache=True)
def _screened_real_components(
    positions, image_translations, eta, cutoff_squared,
):
    count = positions.shape[0]
    result = np.zeros((count, 3), dtype=np.float64)
    eta_squared = eta * eta
    eta_cubic_term = 4.0 * eta * eta * eta / np.sqrt(np.pi)
    gaussian_term = 2.0 * eta / np.sqrt(np.pi)

    for index in prange(count):
        base_x = positions[index, 0]
        base_y = positions[index, 1]
        xx = 0.0
        xy = 0.0
        yy = 0.0
        for image in range(image_translations.shape[0]):
            x = base_x + image_translations[image, 0]
            y = base_y + image_translations[image, 1]
            radius_squared = x * x + y * y
            if radius_squared <= 1.0e-24 or radius_squared > cutoff_squared:
                continue

            radius = np.sqrt(radius_squared)
            inverse_radius_squared = 1.0 / radius_squared
            inverse_radius_cubed = inverse_radius_squared / radius
            ux = x / radius
            uy = y / radius
            exponential = np.exp(-eta_squared * radius_squared)
            radial = (
                math.erfc(eta * radius) * inverse_radius_cubed
                + gaussian_term * exponential * inverse_radius_squared
            )
            longitudinal = eta_cubic_term * exponential

            xx += radial * (3.0 * ux * ux - 1.0) + longitudinal * ux * ux
            xy += radial * (3.0 * ux * uy) + longitudinal * ux * uy
            yy += radial * (3.0 * uy * uy - 1.0) + longitudinal * uy * uy

        result[index, 0] = xx
        result[index, 1] = xy
        result[index, 2] = yy
    return result


@njit(parallel=True, cache=True)
def _direct_periodic_components(positions, image_translations):
    count = positions.shape[0]
    result = np.zeros((count, 3), dtype=np.float64)
    for index in prange(count):
        base_x = positions[index, 0]
        base_y = positions[index, 1]
        xx = 0.0
        xy = 0.0
        yy = 0.0
        for image in range(image_translations.shape[0]):
            x = base_x + image_translations[image, 0]
            y = base_y + image_translations[image, 1]
            radius_squared = x * x + y * y
            if radius_squared <= 1.0e-24:
                continue
            inverse_radius = 1.0 / np.sqrt(radius_squared)
            inverse_radius_cubed = inverse_radius / radius_squared
            ux = x * inverse_radius
            uy = y * inverse_radius
            xx += inverse_radius_cubed * (3.0 * ux * ux - 1.0)
            xy += inverse_radius_cubed * (3.0 * ux * uy)
            yy += inverse_radius_cubed * (3.0 * uy * uy - 1.0)
        result[index, 0] = xx
        result[index, 1] = xy
        result[index, 2] = yy
    return result


def _integer_lattice_vectors(vectors, maximum_norm):
    singular_minimum = np.linalg.svd(vectors, compute_uv=False)[-1]
    extent = int(math.ceil(maximum_norm / singular_minimum)) + 1
    combinations = []
    tolerance = maximum_norm * (1.0 + 1.0e-13)
    for first in range(-extent, extent + 1):
        for second in range(-extent, extent + 1):
            vector = first * vectors[0] + second * vectors[1]
            if np.linalg.norm(vector) <= tolerance:
                combinations.append(vector)
    return np.ascontiguousarray(combinations, dtype=np.float64)


def _real_image_translations(supercell, positions, cutoff):
    maximum_base = float(np.max(np.linalg.norm(positions, axis=1)))
    maximum_translation = cutoff + maximum_base
    singular_minimum = np.linalg.svd(supercell, compute_uv=False)[-1]
    extent = int(math.ceil(maximum_translation / singular_minimum)) + 1
    translations = []
    for first in range(-extent, extent + 1):
        for second in range(-extent, extent + 1):
            translations.append(first * supercell[0] + second * supercell[1])
    return np.ascontiguousarray(translations, dtype=np.float64)


def _reciprocal_components(positions, supercell, eta, cutoff, chunk_size):
    area = abs(np.linalg.det(supercell))
    reciprocal = 2.0 * np.pi * np.linalg.inv(supercell).T
    reciprocal_vectors = _integer_lattice_vectors(reciprocal, cutoff)
    nonzero = np.linalg.norm(reciprocal_vectors, axis=1) > 1.0e-14
    reciprocal_vectors = reciprocal_vectors[nonzero]
    magnitudes = np.linalg.norm(reciprocal_vectors, axis=1)

    erfc_values = np.asarray(
        [math.erfc(value / (2.0 * eta)) for value in magnitudes],
        dtype=np.float64,
    )
    common = -(2.0 * np.pi / area) * erfc_values / magnitudes
    weights = np.column_stack((
        common * reciprocal_vectors[:, 0] * reciprocal_vectors[:, 0],
        common * reciprocal_vectors[:, 0] * reciprocal_vectors[:, 1],
        common * reciprocal_vectors[:, 1] * reciprocal_vectors[:, 1],
    ))

    result = np.empty((positions.shape[0], 3), dtype=np.float64)
    transposed_vectors = reciprocal_vectors.T
    for start in range(0, positions.shape[0], chunk_size):
        stop = min(start + chunk_size, positions.shape[0])
        phases = positions[start:stop] @ transposed_vectors
        result[start:stop] = np.cos(phases) @ weights
    return result, reciprocal_vectors.shape[0]


def _assemble_planar_kernel(components, shape, field_prefactor):
    Ly, Lx = shape
    values = components.reshape(Ly, Lx, 2, 2, 3)
    kernel = np.zeros((Ly, Lx, 2, 3, 2, 3), dtype=np.float64)
    kernel[:, :, :, 0, :, 0] = values[..., 0]
    kernel[:, :, :, 0, :, 1] = values[..., 1]
    kernel[:, :, :, 1, :, 0] = values[..., 1]
    kernel[:, :, :, 1, :, 1] = values[..., 2]
    kernel[:, :, :, 2, :, 2] = -(values[..., 0] + values[..., 2])
    kernel *= field_prefactor
    return kernel


def _symmetrise_kernel(kernel):
    Ly, Lx = kernel.shape[:2]
    for dy in range(Ly):
        partner_y = (-dy) % Ly
        for dx in range(Lx):
            partner_x = (-dx) % Lx
            for target_basis in range(2):
                for source_basis in range(2):
                    first = kernel[dy, dx, target_basis, :, source_basis, :]
                    second = kernel[
                        partner_y, partner_x, source_basis, :, target_basis, :
                    ].T
                    average = 0.5 * (first + second)
                    kernel[dy, dx, target_basis, :, source_basis, :] = average
                    kernel[
                        partner_y, partner_x, source_basis, :, target_basis, :
                    ] = average.T
    return kernel


def dipolar_field_prefactor(magnetic_moment):
    moment = float(magnetic_moment)
    if moment <= 0.0:
        raise ValueError("magnetic_moment must be positive")
    return (
        VACUUM_PERMEABILITY * moment
        / (4.0 * np.pi * ANGSTROM ** 3)
    )


def build_ewald_kernel(
    lattice_vectors_angstrom,
    magnetic_basis_angstrom,
    Lx,
    Ly,
    magnetic_moment,
    tolerance=1.0e-11,
    eta=None,
    real_cutoff_angstrom=None,
    reciprocal_cutoff_inverse_angstrom=None,
    reciprocal_chunk_size=4096,
):
    Lx = int(Lx)
    Ly = int(Ly)
    if Lx < 1 or Ly < 1:
        raise ValueError("Lx and Ly must both be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie between zero and one")
    vectors, basis = _validate_planar_geometry(
        lattice_vectors_angstrom, magnetic_basis_angstrom
    )
    positions_grid = _base_displacements(vectors, basis, Lx, Ly)
    positions = np.ascontiguousarray(positions_grid.reshape(-1, 2))
    supercell = np.ascontiguousarray(
        np.vstack((Lx * vectors[0, :2], Ly * vectors[1, :2]))
    )

    decay_argument = math.sqrt(-math.log(tolerance))
    if eta is not None and real_cutoff_angstrom is None:
        real_cutoff_angstrom = decay_argument / float(eta)
    elif real_cutoff_angstrom is None:
        real_cutoff_angstrom = float(
            min(np.linalg.norm(supercell[0]), np.linalg.norm(supercell[1]))
        )
    real_cutoff_angstrom = float(real_cutoff_angstrom)
    if real_cutoff_angstrom <= 0.0:
        raise ValueError("real_cutoff_angstrom must be positive")
    if eta is None:
        eta = decay_argument / real_cutoff_angstrom
    eta = float(eta)
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if reciprocal_cutoff_inverse_angstrom is None:
        reciprocal_cutoff_inverse_angstrom = 2.0 * eta * decay_argument
    reciprocal_cutoff_inverse_angstrom = float(
        reciprocal_cutoff_inverse_angstrom
    )
    if reciprocal_cutoff_inverse_angstrom <= 0.0:
        raise ValueError("reciprocal cutoff must be positive")

    image_translations = _real_image_translations(
        supercell, positions, real_cutoff_angstrom
    )
    components = _screened_real_components(
        positions,
        image_translations,
        eta,
        real_cutoff_angstrom * real_cutoff_angstrom,
    )
    reciprocal_components, reciprocal_count = _reciprocal_components(
        positions,
        supercell,
        eta,
        reciprocal_cutoff_inverse_angstrom,
        int(reciprocal_chunk_size),
    )
    components += reciprocal_components

    self_sites = np.linalg.norm(positions, axis=1) <= 1.0e-12
    self_correction = 4.0 * eta ** 3 / (3.0 * _SQRT_PI)
    components[self_sites, 0] += self_correction
    components[self_sites, 2] += self_correction

    kernel = _assemble_planar_kernel(
        components, (Ly, Lx), dipolar_field_prefactor(magnetic_moment)
    )
    kernel = _symmetrise_kernel(kernel)
    metadata = {
        "method": "ewald_2d_fft",
        "tolerance": tolerance,
        "eta_inverse_angstrom": eta,
        "real_cutoff_angstrom": real_cutoff_angstrom,
        "reciprocal_cutoff_inverse_angstrom": (
            reciprocal_cutoff_inverse_angstrom
        ),
        "real_image_count": int(image_translations.shape[0]),
        "reciprocal_vector_count": int(reciprocal_count),
    }
    return kernel, metadata


def build_direct_periodic_kernel(
    lattice_vectors_angstrom,
    magnetic_basis_angstrom,
    Lx,
    Ly,
    magnetic_moment,
    image_radius,
):
    vectors, basis = _validate_planar_geometry(
        lattice_vectors_angstrom, magnetic_basis_angstrom
    )
    positions_grid = _base_displacements(vectors, basis, int(Lx), int(Ly))
    positions = np.ascontiguousarray(positions_grid.reshape(-1, 2))
    supercell = np.ascontiguousarray(
        np.vstack((int(Lx) * vectors[0, :2], int(Ly) * vectors[1, :2]))
    )
    radius = int(image_radius)
    if radius < 0:
        raise ValueError("image_radius must be non-negative")
    translations = np.asarray([
        first * supercell[0] + second * supercell[1]
        for first in range(-radius, radius + 1)
        for second in range(-radius, radius + 1)
    ], dtype=np.float64)
    components = _direct_periodic_components(positions, translations)
    kernel = _assemble_planar_kernel(
        components,
        (int(Ly), int(Lx)),
        dipolar_field_prefactor(magnetic_moment),
    )
    return _symmetrise_kernel(kernel)


def build_open_kernel(
    lattice_vectors_angstrom,
    magnetic_basis_angstrom,
    Lx,
    Ly,
    magnetic_moment,
):
    vectors, basis = _validate_planar_geometry(
        lattice_vectors_angstrom, magnetic_basis_angstrom
    )
    grid_y = 2 * int(Ly)
    grid_x = 2 * int(Lx)
    kernel = np.zeros((grid_y, grid_x, 2, 3, 2, 3), dtype=np.float64)
    prefactor = dipolar_field_prefactor(magnetic_moment)
    identity = np.eye(3)
    for dy in range(-(int(Ly) - 1), int(Ly)):
        for dx in range(-(int(Lx) - 1), int(Lx)):
            cell = dx * vectors[0] + dy * vectors[1]
            for target_basis in range(2):
                for source_basis in range(2):
                    displacement = (
                        cell + basis[target_basis] - basis[source_basis]
                    )
                    radius = np.linalg.norm(displacement)
                    if radius <= 1.0e-12:
                        continue
                    direction = displacement / radius
                    kernel[
                        dy % grid_y, dx % grid_x,
                        target_basis, :, source_basis, :,
                    ] = (
                        prefactor / radius ** 3
                        * (3.0 * np.outer(direction, direction) - identity)
                    )
    return kernel


def build_cutoff_kernel(lattice_vectors, magnetic_basis, Lx, Ly, magnetic_moment,
                        cutoff_angstrom):
    vectors, basis = _validate_planar_geometry(lattice_vectors, magnetic_basis)
    cutoff = float(cutoff_angstrom)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("dipolar cutoff must be finite and positive")
    minimum = np.linalg.svd(vectors[:, :2], compute_uv=False)[-1]
    extent = int(np.ceil((cutoff + np.linalg.norm(basis[1] - basis[0])) / minimum)) + 1
    kernel = np.zeros((Ly, Lx, 2, 3, 2, 3), dtype=np.float64)
    prefactor = dipolar_field_prefactor(magnetic_moment)
    interactions = 0
    for target in range(2):
        for source in range(2):
            for dy in range(-extent, extent + 1):
                for dx in range(-extent, extent + 1):
                    r = dx * vectors[0] + dy * vectors[1] + basis[source] - basis[target]
                    squared = np.dot(r, r)
                    if squared <= 1e-24 or squared > cutoff * cutoff * (1 + 1e-14):
                        continue
                    tensor = prefactor * (3 * np.outer(r, r) / squared - np.eye(3)) / squared**1.5
                    kernel[(-dy) % Ly, (-dx) % Lx, target, :, source, :] += tensor
                    interactions += 1
    return kernel, {
        "method": "periodic_radial_cutoff_fft",
        "cutoff_angstrom": cutoff,
        "directed_interactions_per_cell": interactions,
    }


class DipolarFFT:

    def __init__(
        self,
        lattice_vectors_angstrom,
        magnetic_basis_angstrom,
        Lx,
        Ly,
        magnetic_moment,
        pbc_x=True,
        pbc_y=True,
        tolerance=1.0e-11,
        workers=1,
        cutoff_angstrom=None,
    ):
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.magnetic_moment = float(magnetic_moment)
        self.pbc_x = bool(pbc_x)
        self.pbc_y = bool(pbc_y)
        self.cutoff_angstrom = None if cutoff_angstrom is None else float(cutoff_angstrom)
        if self.cutoff_angstrom is not None:
            if not np.isfinite(self.cutoff_angstrom) or self.cutoff_angstrom <= 0:
                raise ValueError("cutoff_angstrom must be finite and positive, or None")
            if not (self.pbc_x and self.pbc_y):
                raise NotImplementedError("radial dipolar cutoff currently requires periodic x and y")
        try:
            self.requested_workers = operator.index(workers)
        except TypeError as error:
            raise ValueError("workers must be a positive integer") from error
        if self.requested_workers < 1:
            raise ValueError("workers must be a positive integer")
        if self.pbc_x != self.pbc_y:
            raise NotImplementedError(
                "dipolar FFT currently supports fully periodic or fully open "
                "boundaries, not mixed periodicity"
            )
        self.grid_shape = (
            (self.Ly, self.Lx) if self.pbc_x
            else (2 * self.Ly, 2 * self.Lx)
        )
        self.fft_workers = (
            self.requested_workers
            if self.grid_shape[0] * self.grid_shape[1] >= 16384 else 1
        )

        vectors, basis = _validate_planar_geometry(
            lattice_vectors_angstrom, magnetic_basis_angstrom
        )
        cache_key = (
            self.Lx, self.Ly, self.pbc_x, self.pbc_y,
            self.cutoff_angstrom,
            float(self.magnetic_moment), float(tolerance),
            vectors.tobytes(), basis.tobytes(),
        )
        cached = _KERNEL_CACHE.get(cache_key)
        if cached is None:
            if self.cutoff_angstrom is not None:
                kernel, metadata = build_cutoff_kernel(
                    vectors, basis, self.Lx, self.Ly,
                    self.magnetic_moment, self.cutoff_angstrom,
                )
            elif self.pbc_x:
                kernel, metadata = build_ewald_kernel(
                    vectors, basis, self.Lx, self.Ly,
                    self.magnetic_moment, tolerance=tolerance,
                )
            else:
                kernel = build_open_kernel(
                    vectors, basis, self.Lx, self.Ly, self.magnetic_moment
                )
                metadata = {
                    "method": "open_zero_padded_fft",
                    "tolerance": 0.0,
                    "eta_inverse_angstrom": 0.0,
                    "real_cutoff_angstrom": 0.0,
                    "reciprocal_cutoff_inverse_angstrom": 0.0,
                    "real_image_count": 0,
                    "reciprocal_vector_count": 0,
                }
            kernel_six = kernel.reshape(*self.grid_shape, 6, 6)
            kernel_fft = scipy_fft.rfft2(
                kernel_six, axes=(0, 1), workers=self.fft_workers
            )
            coefficients = [
                kernel_fft[..., row, column]
                for row in range(6)
                for column in ((2, 5) if row in (2, 5) else (0, 1, 3, 4))
            ]
            out_of_plane = np.array((False, False, True, False, False, True))
            mixed_components = out_of_plane[:, None] != out_of_plane[None, :]
            packed_kernel = (
                None if np.any(kernel_six[..., mixed_components] != 0.0)
                else np.ascontiguousarray(np.stack(coefficients, axis=-1))
            )
            cached = (kernel_fft, packed_kernel, metadata, self.grid_shape)
            _KERNEL_CACHE[cache_key] = cached
        self.kernel_fft, self._packed_kernel_fft, self.metadata, self.grid_shape = cached
        self._field_fft = np.empty(
            self.kernel_fft.shape[:-1], dtype=np.complex128
        )
        self._spin_work = (
            None if self.pbc_x
            else np.zeros((*self.grid_shape, 6), dtype=np.float64)
        )

    def field(self, spins):
        values = np.asarray(spins, dtype=np.float64)
        expected_shape = (2 * self.Lx * self.Ly, 3)
        if values.shape != expected_shape:
            raise ValueError(f"spins must have shape {expected_shape}")
        physical = values.reshape(self.Ly, self.Lx, 6)
        if self.grid_shape == (self.Ly, self.Lx):
            work = physical
        else:
            work = self._spin_work
            work[:self.Ly, :self.Lx] = physical
        spin_fft = scipy_fft.rfft2(
            work, axes=(0, 1), workers=self.fft_workers
        )
        if self._packed_kernel_fft is None:
            np.einsum("...pq,...q->...p", self.kernel_fft, spin_fft,
                      optimize=False, out=self._field_fft)
        else:
            contract = (
                _contract_planar_parallel
                if get_num_threads() > 1 and self.grid_shape[0]*self.grid_shape[1] >= 16384
                else _contract_planar_serial
            )
            contract(self._packed_kernel_fft, spin_fft, self._field_fft)
        field = scipy_fft.irfft2(
            self._field_fft, s=self.grid_shape, axes=(0, 1),
            workers=self.fft_workers,
        )
        return np.ascontiguousarray(
            field[:self.Ly, :self.Lx].reshape(expected_shape)
        )

    def energy(self, spins, field=None):
        values = np.asarray(spins, dtype=np.float64)
        if field is None:
            field = self.field(values)
        return -0.5 * self.magnetic_moment * float(np.sum(values * field))


def evolve_dipolar(
    simulation,
    iterations,
    sample_steps,
    quantum_state=None,
    time_offset=0.0,
    damping_initial=None,
    damping_decay_time=0.0,
    damping_schedule_end_time=0.0,
    record_magnetisation=True,
    record_energy=False,
    record_spins=False,
    fft_thermostat=None,
):
    iterations = int(iterations)
    steps = np.asarray(sample_steps, dtype=np.int64)
    if steps.ndim != 1 or np.any(steps < 0) or np.any(steps >= iterations):
        raise ValueError("sample_steps must contain valid local step indices")
    if steps.size > 1 and np.any(steps[1:] <= steps[:-1]):
        raise ValueError("sample_steps must be strictly increasing")
    if damping_initial is None:
        damping_initial = simulation.lam

    from .damping import damping_at_time, damping_noise_factor, classical_noise_factor
    from .integrators import (
        heun_corrector_shells, heun_corrector_shells_serial,
        heun_predictor_shells, heun_predictor_shells_serial,
        heun_predictor_classical_shells, heun_predictor_quantum_shells,
        heun_corrector_quantum_shells,
    )
    from .rk4 import (
        rk4_rhs_stage, rk4_rhs_stage_serial, rk4_rhs_finish, rk4_rhs_finish_serial,
    )
    from .observables import compute_energy_shells, compute_magnetisation_components
    from .thermostats import (
        classical_noise_field, classical_noise_field_serial,
        quantum_noise_update, quantum_noise_update_serial,
    )

    number_of_sites = simulation.S.shape[0]
    threads = get_num_threads()
    single_thread = threads == 1
    use_rk4 = simulation.integrator == "rk4"
    use_fft = simulation.noise_mode == "quantum_fft"
    if use_fft:
        if fft_thermostat is None:
            fft_thermostat = simulation.fft_thermostat
        if fft_thermostat is None or fft_thermostat.N != number_of_sites:
            raise ValueError("quantum_fft requires an initialized FFT thermostat")
        expected_step = int(round(time_offset / simulation.dt))
        if (not np.isclose(time_offset / simulation.dt, expected_step, rtol=0, atol=1e-6)
                or fft_thermostat.fine_step != expected_step):
            raise ValueError("FFT thermostat clock must match time_offset on the spin grid")
    fused_passes = not use_fft and not use_rk4 and threads >= 8 and number_of_sites >= 32768
    if single_thread or number_of_sites <= 512:
        predictor = heun_predictor_shells_serial
        corrector = heun_corrector_shells_serial
    else:
        predictor = heun_predictor_shells
        corrector = heun_corrector_shells
    if single_thread or number_of_sites <= 128:
        classical_field = classical_noise_field_serial
        quantum_update = quantum_noise_update_serial
    else:
        classical_field = classical_noise_field
        quantum_update = quantum_noise_update
    S = simulation.S
    S_tilde = np.empty_like(S)
    S_new = np.empty_like(S)
    dS1 = np.empty_like(S)
    random_field = np.empty_like(S)
    solver = simulation.dipolar_solver
    zero_field = np.zeros_like(S) if solver is None else None
    field_of = (lambda spins: zero_field) if solver is None else solver.field
    dipolar_field = field_of(S)
    midpoint_field = np.empty_like(S) if use_fft else random_field
    endpoint_field = np.empty_like(S) if use_fft else random_field
    if use_rk4:
        rk4_work = np.empty((3, number_of_sites, 3), dtype=np.float64)
        if single_thread or number_of_sites <= 512:
            rk4_stage_pass = rk4_rhs_stage_serial
            rk4_finish_pass = rk4_rhs_finish_serial
        else:
            rk4_stage_pass = rk4_rhs_stage
            rk4_finish_pass = rk4_rhs_finish
        rk4_common = (
            simulation.row_ptr, simulation.neighbours, simulation.bond_shell,
            simulation.J_mu_shell, simulation.lambda_mu_shell, simulation.D_mu,
            simulation.anisotropy_axis, simulation.field_tesla, simulation.gamma,
        )

    sample_count = len(steps)
    data = {
        "time": (steps.astype(np.float64) + 1.0) * simulation.dt,
    }
    if record_magnetisation:
        data["magnetisation"] = np.empty((sample_count, 4), dtype=np.float64)
    if record_energy:
        data["energy"] = np.empty(sample_count, dtype=np.float64)
        data["dipolar_energy"] = np.empty(sample_count, dtype=np.float64)
    if record_spins:
        data["spin_x"] = np.empty((sample_count, number_of_sites), dtype=np.float32)
        data["spin_y"] = np.empty((sample_count, number_of_sites), dtype=np.float32)
        data["spin_z"] = np.empty((sample_count, number_of_sites), dtype=np.float32)

    if simulation.noise_mode == "quantum":
        if quantum_state is None:
            raise ValueError("quantum_state is required for quantum noise")
        from .thermostats import (
            Gamma5, Gamma6, Omega5, Omega6,
            quantum_noise_field, quantum_noise_field_serial,
        )

        z5, v5, z6, v6 = quantum_state
        quantum_field = (
            quantum_noise_field_serial
            if single_thread or number_of_sites <= 8192 else quantum_noise_field
        )
    elif simulation.noise_mode not in ("classical", "none", "quantum_fft"):
        raise ValueError("noise_mode must be quantum, quantum_fft, classical or none")

    if fused_passes:
        if simulation.noise_mode == "quantum":
            predictor = heun_predictor_quantum_shells
            corrector = heun_corrector_quantum_shells
            predictor_noise_args = (z5, z6)
            corrector_noise_args = (
                simulation.thermostat_rng_states, z5, v5, z6, v6,
                simulation.dt_q, simulation.amp5, simulation.amp6,
                Omega5, Omega6, Gamma5, Gamma6,
            )
        else:
            predictor = heun_predictor_classical_shells
            predictor_noise_args = (simulation.thermostat_rng_states,)

    common = (
        simulation.row_ptr,
        simulation.neighbours,
        simulation.bond_shell,
        simulation.J_mu_shell,
        simulation.lambda_mu_shell,
        simulation.D_mu,
        simulation.anisotropy_axis,
        simulation.field_tesla,
        simulation.dt,
        simulation.gamma,
    )
    next_sample = 0
    spatial = getattr(simulation, "spatial_recorder", None)
    start_step = int(round(time_offset / simulation.dt))
    next_spatial_step = None
    if spatial is not None:
        if not np.isclose(time_offset / simulation.dt, start_step, rtol=0, atol=1e-6):
            raise ValueError("spatial recording requires time_offset on the integration grid")
        spatial.record(start_step, S)
        next_spatial_step = spatial.next_step
    for step in range(iterations):
        step_damping = simulation.lam
        schedule_time = time_offset + step * simulation.dt
        scheduled = (
            damping_decay_time > 0.0
            and (
                damping_schedule_end_time < 0.0
                or schedule_time < damping_schedule_end_time
            )
        )
        if scheduled:
            step_damping = damping_at_time(
                schedule_time,
                float(damping_initial),
                simulation.lam,
                damping_decay_time,
            )

        if use_fft:
            fft_thermostat.fields(step_damping, random_field, midpoint_field, endpoint_field)
        elif simulation.noise_mode == "quantum":
            noise_prefactor = simulation.q_pref
            if scheduled:
                noise_prefactor = (
                    simulation.quantum_noise_scale
                    * damping_noise_factor(step_damping)
                )
            if not fused_passes:
                quantum_field(z5, z6, random_field, noise_prefactor)
        else:
            noise_prefactor = simulation.classical_prefactor
            if scheduled:
                noise_prefactor = (
                    simulation.classical_noise_scale
                    * classical_noise_factor(step_damping)
                )
            if not fused_passes:
                classical_field(
                    simulation.thermostat_rng_states, random_field, noise_prefactor
                )

        if use_rk4:
            k1 = dS1
            k2, k3, k4 = rk4_work
            rk4_stage_pass(S, S, random_field, dipolar_field, k1, S_tilde,
                           *rk4_common, step_damping, 0.5 * simulation.dt)
            dipolar_field_tilde = field_of(S_tilde)
            rk4_stage_pass(S, S_tilde, midpoint_field, dipolar_field_tilde, k2, S_new,
                           *rk4_common, step_damping, 0.5 * simulation.dt)
            dipolar_field_tilde = field_of(S_new)
            rk4_stage_pass(S, S_new, midpoint_field, dipolar_field_tilde, k3, S_tilde,
                           *rk4_common, step_damping, simulation.dt)
            dipolar_field_tilde = field_of(S_tilde)
            rk4_finish_pass(S, S_tilde, endpoint_field, dipolar_field_tilde,
                            k1, k2, k3, k4, S_new, *rk4_common, step_damping,
                            simulation.dt)
        else:
            if fused_passes:
                predictor(
                    S, S_tilde, dS1, random_field, dipolar_field, *common,
                    step_damping, *predictor_noise_args, noise_prefactor,
                )
            else:
                predictor(
                    S, S_tilde, dS1, random_field, dipolar_field, *common,
                    step_damping,
                )
            dipolar_field_tilde = field_of(S_tilde)
            if fused_passes and simulation.noise_mode == "quantum":
                corrector(
                    S, S_tilde, S_new, dS1, random_field, dipolar_field_tilde,
                    *common, step_damping, *corrector_noise_args,
                )
            else:
                corrector(
                    S, S_tilde, S_new, dS1, endpoint_field, dipolar_field_tilde,
                    *common, step_damping,
                )
        S, S_new = S_new, S
        if use_fft:
            fft_thermostat.advance()

        if simulation.noise_mode == "quantum" and not fused_passes:
            quantum_update(
                simulation.thermostat_rng_states,
                z5, v5, z6, v6,
                simulation.dt_q, simulation.amp5, simulation.amp6,
                Omega5, Omega6, Gamma5, Gamma6,
            )

        if next_spatial_step is not None and start_step + step + 1 == next_spatial_step:
            spatial.record(next_spatial_step, S)
            next_spatial_step = spatial.next_step

        is_sample = next_sample < sample_count and step == steps[next_sample]
        need_next_field = step + 1 < iterations
        if need_next_field or (is_sample and record_energy):
            dipolar_field = field_of(S)

        if is_sample:
            if record_magnetisation:
                data["magnetisation"][next_sample] = (
                    compute_magnetisation_components(S)
                )
            if record_energy:
                local_energy = compute_energy_shells(
                    S,
                    simulation.row_ptr,
                    simulation.neighbours,
                    simulation.bond_shell,
                    simulation.J_shell,
                    simulation.lambda_shell,
                    simulation.D,
                    simulation.anisotropy_axis,
                    simulation.field_tesla,
                    simulation.magnetic_moment,
                )
                dipolar_energy = 0.0 if solver is None else solver.energy(S, dipolar_field)
                data["dipolar_energy"][next_sample] = dipolar_energy
                data["energy"][next_sample] = local_energy + dipolar_energy
            if record_spins:
                data["spin_x"][next_sample] = S[:, 0]
                data["spin_y"][next_sample] = S[:, 1]
                data["spin_z"][next_sample] = S[:, 2]
            next_sample += 1

    simulation.S = S
    return data
