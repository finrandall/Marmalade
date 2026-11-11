import argparse
import importlib.util
import operator
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "core"

from .constants import MILLIELECTRON_VOLT
from .observables import compute_energy
from .simulation import Simulation

THETA_MIN_DEGREES = 90.0
THETA_MAX_DEGREES = 90.0
THETA_POINTS = 1

PHI_MIN_DEGREES = 0.0
PHI_MAX_DEGREES = 360.0
PHI_POINTS = 3600

PLOT_RELATIVE_ENERGY = True
SHOW_PLOT = True
ENERGY_TOLERANCE_MEV_PER_SITE = 1.0e-10


def _single_parameters(params):
    selected = {}
    for name, value in params.items():
        if isinstance(value, np.ndarray):
            if name in ("anisotropy_axis", "field_tesla") and value.shape == (3,):
                selected[name] = value.copy()
                continue
            if value.ndim > 0:
                if len(value) != 1:
                    raise ValueError(f"select one value for parameter {name!r}")
                value = value[0]
            if np.ndim(value) == 0:
                value = np.asarray(value).item()
        selected[name] = value
    return selected


def load_parameters(path=None):
    if path is None:
        from .parameters import default_parameters
        return _single_parameters(default_parameters()[0])
    path = Path(path)
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("_asd_ground_state_parameters", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load parameters from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    defaults = module.default_parameters()
    params = defaults[0] if isinstance(defaults, tuple) else defaults
    return _single_parameters(params)


def spin_direction(theta_degrees, phi_degrees):
    if theta_degrees == 0.0:
        return np.array((0.0, 0.0, 1.0))
    if theta_degrees == 180.0:
        return np.array((0.0, 0.0, -1.0))
    theta, phi = np.deg2rad((theta_degrees, phi_degrees))
    direction = np.array((
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        0.0 if theta_degrees == 90.0 else np.cos(theta),
    ))
    return direction / np.linalg.norm(direction)


def angular_grid(minimum, maximum, points, name):
    minimum, maximum = float(minimum), float(maximum)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{name} bounds must be finite")
    if isinstance(points, (bool, np.bool_)) or not isinstance(
        points, (int, np.integer),
    ):
        raise ValueError(f"{name} points must be a positive integer")
    if points < 1:
        raise ValueError(f"{name} points must be a positive integer")
    if maximum < minimum:
        raise ValueError(f"{name} maximum must be at least its minimum")
    if name == "theta" and not 0.0 <= minimum <= maximum <= 180.0:
        raise ValueError("theta bounds must lie between 0 and 180 degrees")
    span = maximum - minimum
    if name == "phi" and span > 360.0:
        raise ValueError("phi range must span at most 360 degrees")
    if points == 1:
        if minimum != maximum:
            raise ValueError(f"for one {name} point, set its minimum and maximum equal")
        return np.array([minimum])
    if minimum == maximum:
        raise ValueError(f"for a fixed {name}, use one point")
    full_turn = name == "phi" and np.isclose(span, 360.0, rtol=0.0, atol=1.0e-10)
    return np.linspace(minimum, maximum, points, endpoint=not full_turn)


def sweep_ferromagnetic_ground_state(
    params=None, *,
    theta_min_degrees=THETA_MIN_DEGREES,
    theta_max_degrees=THETA_MAX_DEGREES,
    theta_points=THETA_POINTS,
    phi_min_degrees=PHI_MIN_DEGREES,
    phi_max_degrees=PHI_MAX_DEGREES,
    phi_points=PHI_POINTS,
    energy_tolerance_meV_per_site=ENERGY_TOLERANCE_MEV_PER_SITE,
    energy_function=None,
    number_of_spins=None,
):
    theta_degrees = angular_grid(
        theta_min_degrees, theta_max_degrees, theta_points, "theta",
    )
    phi_degrees = angular_grid(
        phi_min_degrees, phi_max_degrees, phi_points, "phi",
    )
    tolerance = float(energy_tolerance_meV_per_site)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("energy tolerance must be finite and non-negative")

    if energy_function is None:
        if number_of_spins is not None:
            raise ValueError("number_of_spins requires an energy_function returning total joules")
        params = load_parameters() if params is None else _single_parameters(params)
        simulation = Simulation(params)
        simulation.initialise(include_structure=False)
        number_of_spins = simulation.N
        shells = simulation.bond_shell
        energy_arguments = (
            simulation.row_ptr, simulation.neighbours,
            simulation.J_shell[shells], simulation.lambda_shell[shells],
            simulation.D, simulation.anisotropy_axis,
            simulation.field_tesla, simulation.magnetic_moment,
        )

        def energy_function(spins):
            energy = compute_energy(spins, *energy_arguments)
            if simulation.dipolar_solver is not None:
                energy += simulation.dipolar_solver.energy(spins)
            return energy
    else:
        if not callable(energy_function):
            raise TypeError("energy_function must be callable and return total joules")
        try:
            count = operator.index(number_of_spins)
        except TypeError:
            raise ValueError("number_of_spins must be a positive integer") from None
        if isinstance(number_of_spins, (bool, np.bool_)) or count < 1:
            raise ValueError("number_of_spins must be a positive integer")
        number_of_spins = count
        params = {} if params is None else dict(params)

    energies = np.empty((len(theta_degrees), len(phi_degrees)))
    spins = np.empty((number_of_spins, 3), dtype=np.float64)
    for i, theta in enumerate(theta_degrees):
        for j, phi in enumerate(phi_degrees):
            spins[:] = spin_direction(theta, phi)
            energies[i, j] = float(energy_function(spins)) / (
                number_of_spins * MILLIELECTRON_VOLT
            )
    if not np.all(np.isfinite(energies)):
        raise ValueError("non-finite energies: check the Hamiltonian parameters")

    minimum_index = np.unravel_index(np.argmin(energies), energies.shape)
    minimum_energy = float(energies[minimum_index])
    tied_minima = energies - minimum_energy <= tolerance
    pole_rows = (theta_degrees == 0.0) | (theta_degrees == 180.0)
    tied_minima[pole_rows, 1:] = False
    tied_indices = np.argwhere(tied_minima)
    tied_angles = np.column_stack((
        theta_degrees[tied_indices[:, 0]], phi_degrees[tied_indices[:, 1]],
    ))
    theta_min = float(theta_degrees[minimum_index[0]])
    phi_min = float(phi_degrees[minimum_index[1]])
    return {
        "parameters": params,
        "number_of_spins": number_of_spins,
        "theta_degrees": theta_degrees,
        "phi_degrees": phi_degrees,
        "energy_meV_per_site": energies,
        "minimum_energy_meV_per_site": minimum_energy,
        "minimum_theta_degrees": theta_min,
        "minimum_phi_degrees": phi_min,
        "minimum_direction": spin_direction(theta_min, phi_min),
        "energy_tolerance_meV_per_site": tolerance,
        "tied_minimum_angles_degrees": tied_angles,
    }


def plot_energy_vs_phi(result, relative_energy=PLOT_RELATIVE_ENERGY):
    import matplotlib.pyplot as plt

    phi = result["phi_degrees"]
    theta = result["theta_degrees"]
    energies = result["energy_meV_per_site"]
    plotted_energies = (
        energies - result["minimum_energy_meV_per_site"]
        if relative_energy else energies
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for angle, row in zip(theta, plotted_energies):
        axis.plot(phi, row, label=rf"$\theta = {angle:g}^\circ$")
    axis.set_xlabel(r"$\phi$ (degrees)")
    axis.set_ylabel(
        r"$E - E_{\min}$ (meV/site)" if relative_energy else "Energy (meV/site)"
    )
    axis.set_title("Uniform ferromagnetic energy versus azimuth")
    axis.grid(alpha=0.25)
    if len(theta) <= 10:
        axis.legend()
    else:
        axis.text(
            0.02, 0.98,
            f"{len(theta)} curves: theta = {theta[0]:g} to {theta[-1]:g} degrees",
            transform=axis.transAxes, va="top",
        )
    if len(phi) == 1:
        for line in axis.lines:
            line.set_marker("o")
    else:
        axis.set_xlim(phi[0], phi[-1])

    tolerance = result["energy_tolerance_meV_per_site"]
    if np.ptp(energies) <= tolerance:
        centre = float(np.mean(plotted_energies))
        padding = max(
            10.0 * tolerance, 1.0e-12 if relative_energy else 1.0e-6,
        )
        axis.set_ylim(centre - padding, centre + padding)
        axis.text(
            0.5, 0.05,
            f"All sampled energies are equal within {tolerance:g} meV/site",
            transform=axis.transAxes, ha="center",
        )
    if not relative_energy:
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    return figure


def main(params=None, args=None):
    parser = argparse.ArgumentParser(
        description="Scan uniformly aligned spin directions using the selected system's Hamiltonian."
    )
    parser.add_argument("--parameters-file", type=Path)
    parser.add_argument("--theta-min", type=float, default=THETA_MIN_DEGREES)
    parser.add_argument("--theta-max", type=float, default=THETA_MAX_DEGREES)
    parser.add_argument("--theta-points", type=int, default=THETA_POINTS)
    parser.add_argument("--phi-min", type=float, default=PHI_MIN_DEGREES)
    parser.add_argument("--phi-max", type=float, default=PHI_MAX_DEGREES)
    parser.add_argument("--phi-points", type=int, default=PHI_POINTS)
    parser.add_argument("--energy-tolerance", type=float, default=ENERGY_TOLERANCE_MEV_PER_SITE)
    parser.add_argument("--no-plot", action="store_true")
    options = parser.parse_args(args)
    if params is not None and options.parameters_file is not None:
        parser.error("supply params or --parameters-file, rather than both")
    if params is None:
        params = load_parameters(options.parameters_file)
    result = sweep_ferromagnetic_ground_state(
        params,
        theta_min_degrees=options.theta_min,
        theta_max_degrees=options.theta_max,
        theta_points=options.theta_points,
        phi_min_degrees=options.phi_min,
        phi_max_degrees=options.phi_max,
        phi_points=options.phi_points,
        energy_tolerance_meV_per_site=options.energy_tolerance,
    )
    print(f"Spin sites: {result['number_of_spins']}")
    print(f"Lattice: {result['parameters'].get('lattice', 'cubic')}")
    print(f"Angular grid: {result['energy_meV_per_site'].shape} (theta, phi)")
    print(f"Lowest sampled FM energy: {result['minimum_energy_meV_per_site']:.12f} meV/site")
    print(
        f"One minimum: theta={result['minimum_theta_degrees']:g} degrees, "
        f"phi={result['minimum_phi_degrees']:g} degrees"
    )
    print(f"Common spin direction: {result['minimum_direction']}")
    angles = result["tied_minimum_angles_degrees"]
    print(
        f"Sampled directions tied within "
        f"{result['energy_tolerance_meV_per_site']:g} meV/site: {len(angles)}"
    )
    if SHOW_PLOT and not options.no_plot:
        plot_energy_vs_phi(result)
        import matplotlib.pyplot as plt
        plt.show()
    return result


if __name__ == "__main__":
    main()
