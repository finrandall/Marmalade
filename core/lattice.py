import numpy as np
from numba import njit

from .structure import (
    REFERENCE_LATTICE_CONSTANT_ANGSTROM,
    cubic_basis,
    monolayer_basis,
)


SHELL_NAMES = ("J1", "J2", "J3")

HONEYCOMB_BOND_TEMPLATES = (
    (0, 1, 0, 0, 0),
    (0, 1, -1, 0, 0),
    (0, 1, -1, -1, 0),
    (0, 0, 1, 0, 1),
    (0, 0, 0, 1, 1),
    (0, 0, 1, 1, 1),
    (1, 1, 1, 0, 1),
    (1, 1, 0, 1, 1),
    (1, 1, 1, 1, 1),
    (0, 1, -2, -1, 2),
    (0, 1, 0, 1, 2),
    (0, 1, 0, -1, 2),
)

_BOND_TEMPLATES = np.asarray(HONEYCOMB_BOND_TEMPLATES, dtype=np.int32)
_SPIN_GRAPH_CACHE = {}


def site_index(ix, iy, basis, Lx):
    return basis + 2 * (ix + Lx * iy)


@njit(cache=True)
def _build_csr_arrays(Lx, Ly, pbc_x, pbc_y, vectors, magnetic_basis,
                      include_displacements):
    number_of_sites = 2 * Lx * Ly
    counts = np.zeros(number_of_sites, dtype=np.int32)

    for iy in range(Ly):
        for ix in range(Lx):
            for template_index in range(_BOND_TEMPLATES.shape[0]):
                source_basis = _BOND_TEMPLATES[template_index, 0]
                target_basis = _BOND_TEMPLATES[template_index, 1]
                dx = _BOND_TEMPLATES[template_index, 2]
                dy = _BOND_TEMPLATES[template_index, 3]

                target_x = ix + dx
                target_y = iy + dy
                if pbc_x:
                    target_x %= Lx
                elif target_x < 0 or target_x >= Lx:
                    continue
                if pbc_y:
                    target_y %= Ly
                elif target_y < 0 or target_y >= Ly:
                    continue

                source = source_basis + 2 * (ix + Lx * iy)
                target = target_basis + 2 * (target_x + Lx * target_y)
                counts[source] += 1
                counts[target] += 1

    row_ptr = np.empty(number_of_sites + 1, dtype=np.int64)
    row_ptr[0] = 0
    for site in range(number_of_sites):
        row_ptr[site + 1] = row_ptr[site] + counts[site]

    number_of_edges = row_ptr[number_of_sites]
    neighbours = np.empty(number_of_edges, dtype=np.int32)
    bond_shell = np.empty(number_of_edges, dtype=np.int8)
    if include_displacements:
        bond_displacement = np.empty((number_of_edges, 3), dtype=np.float64)
    else:
        bond_displacement = np.empty((0, 3), dtype=np.float64)
    cursor = row_ptr[:-1].copy()

    for iy in range(Ly):
        for ix in range(Lx):
            for template_index in range(_BOND_TEMPLATES.shape[0]):
                source_basis = _BOND_TEMPLATES[template_index, 0]
                target_basis = _BOND_TEMPLATES[template_index, 1]
                dx = _BOND_TEMPLATES[template_index, 2]
                dy = _BOND_TEMPLATES[template_index, 3]
                shell = _BOND_TEMPLATES[template_index, 4]

                target_x = ix + dx
                target_y = iy + dy
                if pbc_x:
                    target_x %= Lx
                elif target_x < 0 or target_x >= Lx:
                    continue
                if pbc_y:
                    target_y %= Ly
                elif target_y < 0 or target_y >= Ly:
                    continue

                source = source_basis + 2 * (ix + Lx * iy)
                target = target_basis + 2 * (target_x + Lx * target_y)
                source_edge = cursor[source]
                target_edge = (
                    source_edge + 1 if source == target else cursor[target]
                )
                neighbours[source_edge] = target
                neighbours[target_edge] = source
                bond_shell[source_edge] = shell
                bond_shell[target_edge] = shell
                if include_displacements:
                    for component in range(3):
                        displacement = (
                            magnetic_basis[target_basis, component]
                            - magnetic_basis[source_basis, component]
                            + dx * vectors[0, component]
                            + dy * vectors[1, component]
                        )
                        bond_displacement[source_edge, component] = displacement
                        bond_displacement[target_edge, component] = -displacement
                cursor[source] += 1
                cursor[target] += 1
    return row_ptr, neighbours, bond_shell, bond_displacement


def build_honeycomb_monolayer(
        Lx, Ly,
        lattice_constant_angstrom=REFERENCE_LATTICE_CONSTANT_ANGSTROM,
        pbc_x=True, pbc_y=True, include_structure=True):
    Lx = int(Lx)
    Ly = int(Ly)
    if Lx < 1 or Ly < 1:
        raise ValueError("Lx and Ly must both be at least one")
    vectors, magnetic_basis, cr_neighbours = monolayer_basis(
        lattice_constant_angstrom)
    lattice_constant_angstrom = float(lattice_constant_angstrom)
    cache_key = (
        Lx, Ly, bool(pbc_x), bool(pbc_y), bool(include_structure),
        lattice_constant_angstrom,
    )
    if not include_structure and cache_key in _SPIN_GRAPH_CACHE:
        return _SPIN_GRAPH_CACHE[cache_key]

    basis_fractional = (
        magnetic_basis[:, :2] @ np.linalg.inv(vectors[:, :2])
    )

    number_of_sites = 2 * Lx * Ly
    basis_indices = (np.arange(number_of_sites, dtype=np.int32) % 2).astype(
        np.int8
    )
    row_ptr, neighbours, bond_shell, bond_displacement = _build_csr_arrays(
        Lx, Ly, bool(pbc_x), bool(pbc_y), vectors, magnetic_basis,
        bool(include_structure),
    )

    supercell_vectors = vectors.copy()
    supercell_vectors[0] *= Lx
    supercell_vectors[1] *= Ly

    if include_structure:
        positions = np.empty((number_of_sites, 3), dtype=np.float64)
        cell_indices = np.empty((number_of_sites, 2), dtype=np.int32)
        for iy in range(Ly):
            for ix in range(Lx):
                translation = ix * vectors[0] + iy * vectors[1]
                for basis in range(2):
                    index = site_index(ix, iy, basis, Lx)
                    positions[index] = translation + magnetic_basis[basis]
                    cell_indices[index] = (ix, iy)
        atomic_species = np.full(number_of_sites, "Cr")
        magnetic_atom_indices = np.arange(number_of_sites, dtype=np.int64)
    else:
        positions = np.empty((0, 3), dtype=np.float64)
        cell_indices = np.empty((0, 2), dtype=np.int32)
        atomic_species = np.empty(0, dtype="U2")
        magnetic_atom_indices = np.empty(0, dtype=np.int64)

    lattice = {
        "lattice_constant_angstrom": lattice_constant_angstrom,
        "lattice_vectors_angstrom": vectors,
        "supercell_vectors_angstrom": supercell_vectors,
        "basis_fractional": basis_fractional,
        "magnetic_basis_angstrom": magnetic_basis,
        "positions_angstrom": positions,
        "atomic_positions_angstrom": positions,
        "atomic_species": atomic_species,
        "magnetic_atom_indices": magnetic_atom_indices,
        "atomic_basis_angstrom": magnetic_basis,
        "nearest_cr_displacements_angstrom": cr_neighbours,
        "cell_indices": cell_indices,
        "basis_indices": basis_indices,
        "row_ptr": row_ptr,
        "neighbours": neighbours,
        "bond_shell": bond_shell,
        "bond_displacement_angstrom": bond_displacement,
    }
    if not include_structure:
        _SPIN_GRAPH_CACHE[cache_key] = lattice
    return lattice


def active_bond_graph(row_ptr, neighbours, bond_shell, J_shell, lambda_shell):
    J_shell = np.asarray(J_shell)
    lambda_shell = np.asarray(lambda_shell)
    active_shell = (J_shell != 0.0) | (lambda_shell != 0.0)
    keep = active_shell[bond_shell]
    if keep.all():
        return row_ptr, neighbours, bond_shell
    number_of_sites = len(row_ptr) - 1
    edge_site = np.repeat(
        np.arange(number_of_sites, dtype=np.int64), np.diff(row_ptr)
    )
    counts = np.bincount(edge_site[keep], minlength=number_of_sites)
    active_row_ptr = np.zeros(number_of_sites + 1, dtype=row_ptr.dtype)
    np.cumsum(counts, out=active_row_ptr[1:])
    return (
        active_row_ptr,
        np.ascontiguousarray(neighbours[keep]),
        np.ascontiguousarray(bond_shell[keep]),
    )


def bond_statistics(lattice):
    number_of_sites = len(lattice["basis_indices"])
    directed = np.bincount(lattice["bond_shell"], minlength=len(SHELL_NAMES))
    return {
        "number_of_sites": number_of_sites,
        "directed_bonds": dict(zip(SHELL_NAMES, directed.tolist())),
        "undirected_bonds": dict(zip(SHELL_NAMES, (directed // 2).tolist())),
        "mean_coordination": dict(
            zip(SHELL_NAMES, (directed / number_of_sites).tolist())
        ),
    }


@njit(cache=True)
def _build_cubic_arrays(Lx, Ly, Lz, pbc_x, pbc_y, pbc_z,
                        spacing, include_displacements):
    count = Lx * Ly * Lz
    dimensions = (Lx, Ly, Lz)
    periodic = (pbc_x, pbc_y, pbc_z)
    strides = (1, Lx, Lx * Ly)
    row_ptr = np.empty(count + 1, dtype=np.int64)
    row_ptr[0] = 0
    for site in range(count):
        coordinates = (site % Lx, (site // Lx) % Ly, site // (Lx * Ly))
        degree = 0
        for axis in range(3):
            for direction in (-1, 1):
                target = coordinates[axis] + direction
                if periodic[axis] or 0 <= target < dimensions[axis]:
                    degree += 1
        row_ptr[site + 1] = row_ptr[site] + degree
    neighbours = np.empty(row_ptr[-1], dtype=np.int32)
    displacement_count = row_ptr[-1] if include_displacements else 0
    displacements = np.zeros((displacement_count, 3), dtype=np.float64)
    for site in range(count):
        coordinates = (site % Lx, (site // Lx) % Ly, site // (Lx * Ly))
        edge = row_ptr[site]
        for axis in range(3):
            for direction in (-1, 1):
                target = coordinates[axis] + direction
                if periodic[axis]:
                    target %= dimensions[axis]
                elif target < 0 or target >= dimensions[axis]:
                    continue
                neighbours[edge] = site + (target - coordinates[axis]) * strides[axis]
                if include_displacements:
                    displacements[edge, axis] = direction * spacing
                edge += 1
    return row_ptr, neighbours, displacements


def build_simple_cubic(Lx, Ly, Lz, lattice_constant_angstrom=1.0,
                       pbc_x=True, pbc_y=True, pbc_z=True,
                       include_structure=True):
    Lx, Ly, Lz = int(Lx), int(Ly), int(Lz)
    spacing = float(lattice_constant_angstrom)
    if min(Lx, Ly, Lz) < 1:
        raise ValueError("Lx, Ly and Lz must all be at least one")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice_constant_angstrom must be finite and positive")
    count = Lx * Ly * Lz
    if count > np.iinfo(np.int32).max:
        raise ValueError("the cubic lattice exceeds the supported site-index range")
    row_ptr, neighbours, displacements = _build_cubic_arrays(
        Lx, Ly, Lz, bool(pbc_x), bool(pbc_y), bool(pbc_z),
        spacing, bool(include_structure),
    )
    sites = np.arange(count, dtype=np.int64)
    parity = ((sites % Lx + (sites // Lx) % Ly + sites // (Lx * Ly)) % 2).astype(np.int8)
    if include_structure:
        cell_indices = np.column_stack((
            sites % Lx, (sites // Lx) % Ly, sites // (Lx * Ly),
        )).astype(np.int32)
        positions = cell_indices.astype(np.float64) * spacing
        species = np.full(count, "spin")
        magnetic_atoms = sites
    else:
        cell_indices = np.empty((0, 3), dtype=np.int32)
        positions = np.empty((0, 3), dtype=np.float64)
        species = np.empty(0, dtype="U4")
        magnetic_atoms = np.empty(0, dtype=np.int64)
    vectors, basis, nearest = cubic_basis(spacing)
    return {
        "lattice_constant_angstrom": spacing,
        "lattice_vectors_angstrom": vectors,
        "supercell_vectors_angstrom": np.diag(np.array((Lx, Ly, Lz)) * spacing),
        "basis_fractional": basis,
        "magnetic_basis_angstrom": basis,
        "positions_angstrom": positions,
        "atomic_positions_angstrom": positions,
        "atomic_species": species,
        "magnetic_atom_indices": magnetic_atoms,
        "atomic_basis_angstrom": basis,
        "nearest_neighbour_displacements_angstrom": nearest,
        "cell_indices": cell_indices,
        "basis_indices": np.zeros(count, dtype=np.int8),
        "sublattice_indices": parity,
        "row_ptr": row_ptr,
        "neighbours": neighbours,
        "bond_shell": np.zeros(len(neighbours), dtype=np.int8),
        "bond_displacement_angstrom": displacements,
    }
