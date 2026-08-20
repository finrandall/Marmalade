import csv
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import get_num_threads, set_num_threads

from parameters import get_parameters
from simulation import Simulation


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def format_elapsed(seconds):
    return f"{int(seconds // 60):02}:{seconds % 60:05.2f}"


def configure_threads(settings):
    available_threads = get_num_threads()
    threads_per_rank = settings.get("threads_per_rank")
    if threads_per_rank is None:
        threads_per_rank = max(1, available_threads // size)
    if threads_per_rank < 1 or threads_per_rank > available_threads:
        raise ValueError(
            f"threads_per_rank must be between 1 and {available_threads}")
    set_num_threads(threads_per_rank)
    if rank == 0:
        print(f"MPI ranks: {size}; Numba threads per rank: {threads_per_rank}")


def parameter_combinations(params):
    names = tuple(params)
    return [dict(zip(names, values)) for values in itertools.product(*(params[name] for name in names))]


def validate_combination(params, settings):
    if not settings["outputs"]:
        raise ValueError("At least one output must be selected")
    if params["integrator"] != "heun":
        raise ValueError("ASD currently supports only the Heun integrator")
    if params["noise_mode"] == "quantum" and params["integrator"] != "heun":
        raise ValueError("Quantum noise currently only supports the Heun integrator")
    if "spin_trajectory" in settings["outputs"] and params["integrator"] != "heun":
        raise ValueError("Spin trajectory output currently only supports the Heun integrator")
    if params["dt"] <= 0.0 or params["end_time"] <= 0.0:
        raise ValueError("dt and end_time must be positive")
    if params["burn_in_time"] < 0.0 or params["burn_in_time"] >= params["end_time"]:
        raise ValueError("burn_in_time must be non-negative and smaller than end_time")
    if params["stride"] < 1:
        raise ValueError("stride must be at least 1")
    if min(params["Lx"], params["Ly"], params["Lz"]) < 1:
        raise ValueError("lattice dimensions must be at least 1")


def partition_combinations(combinations, settings):
    valid = []
    rejected = []
    for combination in combinations:
        try:
            validate_combination(combination, settings)
        except ValueError as error:
            rejected.append((combination, str(error)))
        else:
            valid.append(combination)
    return valid, rejected


def save_results(results, settings):
    output_directory = Path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_directory / timestamp
    suffix = 1
    while output_path.exists():
        output_path = output_directory / f"{timestamp}_{suffix:02}"
        suffix += 1
    output_path.mkdir()
    index_rows = []
    for i, result in enumerate(results):
        filename = f"simulation_{i:04}.npz"
        archive = {**result["parameters"], "outputs": np.asarray(settings["outputs"]), **result["data"]}
        np.savez_compressed(output_path / filename, **archive)
        index_rows.append({"file": filename, "outputs": ";".join(settings["outputs"]), **result["parameters"]})
    with open(output_path / "index.csv", "w", newline="") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    return output_path


def main(params=None, settings=None):
    if rank == 0 and params is None:
        params, settings = get_parameters()
    if rank != 0:
        params = settings = None
    params = comm.bcast(params, root=0)
    settings = comm.bcast(settings, root=0)
    configure_threads(settings)

    if rank == 0:
        combinations, rejected = partition_combinations(parameter_combinations(params), settings)
        for combination, reason in rejected:
            choices = ", ".join(f"{name}={combination[name]}" for name in ("noise_mode", "integrator", "initial_state"))
            print(f"Skipping incompatible combination ({choices}): {reason}")
        if not combinations:
            raise ValueError("No valid parameter combinations remain")
    else:
        combinations = None
    combinations = comm.bcast(combinations, root=0)

    start = time.time()
    local_results = []
    for i in range(rank, len(combinations), size):
        point_started = time.time()
        data = Simulation(combinations[i], settings["outputs"]).run()
        point_elapsed = time.time() - point_started
        combination = combinations[i]
        print(
            f"Point {i + 1}/{len(combinations)}: T={combination['T']:g} K, "
            f"L=({combination['Lx']}, {combination['Ly']}, {combination['Lz']}) "
            f"completed on rank {rank} in {format_elapsed(point_elapsed)}",
            flush=True,
        )
        local_results.append((i, {"parameters": combinations[i], "data": data}))
    gathered = comm.gather(local_results, root=0)
    elapsed = time.time() - start
    if rank != 0:
        return None

    indexed = [item for process_results in gathered for item in process_results]
    indexed.sort(key=lambda item: item[0])
    results = [result for _, result in indexed]
    print(f"Simulation time: {format_elapsed(elapsed)}")
    print(f"Completed simulations: {len(results)}")
    output_path = save_results(results, settings)
    print(f"Saved results to: {output_path}")
    return results


if __name__ == "__main__":
    main()
