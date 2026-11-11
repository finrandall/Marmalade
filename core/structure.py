import numpy as np


REFERENCE_LATTICE_CONSTANT_ANGSTROM = 1.0


def cubic_basis(lattice_constant_angstrom=REFERENCE_LATTICE_CONSTANT_ANGSTROM):
    a = float(lattice_constant_angstrom)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("lattice_constant_angstrom must be finite and positive")
    vectors = np.eye(3) * a
    basis = np.zeros((1, 3), dtype=np.float64)
    neighbours = np.concatenate((vectors, -vectors))
    return vectors, basis, neighbours


def monolayer_basis(lattice_constant_angstrom=REFERENCE_LATTICE_CONSTANT_ANGSTROM):
    a = float(lattice_constant_angstrom)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("lattice_constant_angstrom must be finite and positive")
    lattice_vectors = np.array(
        ((a, 0.0, 0.0), (-0.5 * a, np.sqrt(3.0) * a / 2.0, 0.0)),
        dtype=np.float64,
    )
    magnetic_basis = np.array(
        ((0.0, 0.0, 0.0), (0.5 * a, a / (2.0 * np.sqrt(3.0)), 0.0)),
        dtype=np.float64,
    )
    cr_neighbours = np.array(
        (magnetic_basis[1],
         magnetic_basis[1] - lattice_vectors[0],
         magnetic_basis[1] - lattice_vectors[0] - lattice_vectors[1]),
        dtype=np.float64,
    )
    return lattice_vectors, magnetic_basis, cr_neighbours
