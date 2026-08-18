# ASD

ASD is a reusable Python toolbox for atomistic spin dynamics using the
Landau-Lifshitz-Gilbert equation. Research projects can import individual
numerical components or use the optional high-level simulation interface.

## Package structure

The reusable modules live directly in the `ASD` folder:

- `integrators.py`: Heun and stochastic rotation routines
- `llg.py`: Landau-Lifshitz-Gilbert equation
- `noise.py`: classical and quantum noise
- `hamiltonian.py`: Hamiltonian and effective-field calculations
- `lattice.py`: lattice and neighbour-list builders
- `initialise.py`: reusable initial conditions
- `evolve.py`: deterministic and stochastic evolution routines
- `magnetisation.py` and `spectrum.py`: reusable observables and analysis
- `simulation.py`: optional high-level simulation interface

A specialised research project can import only the machinery it needs:

```python
from integrators import heun_step
from llg import llg_rhs
```

Open the ASD folder as the PyCharm project. Project-specific scripts live under
`Projects` and can import the shared modules from the ASD folder.

## High-level simulations

`main.py` and `parameters.py` provide the existing sweep-oriented convenience
workflow. A one-element parameter array fixes a parameter; multiple elements
sweep it. Command-line options override those arrays:

```bash
python main.py --temperature 10 30 100 --noise-mode classical none
mpiexec -n 4 python main.py --temperature 10 30 100
```

Stored data is selected independently:

```bash
python main.py --outputs magnetisation_mean
python main.py --outputs magnetisation_timeseries
python main.py --outputs spin_trajectory
```

`magnetisation_mean` accumulates post-burn-in means, standard errors and a sample
count without allocating a full time series. `magnetisation_timeseries` stores the
sampled magnetisation history. `spin_trajectory` stores the sampled component of
every spin.

Results are written as compressed `.npz` archives with an `index.csv`. Simulation
code does not plot automatically; research projects can use `magnetisation.py`, `spectrum.py`,
or their own analysis scripts.

## Development boundary

ASD contains numerical machinery that is reusable across ASD projects.
Experiment-specific parameter choices, analytic benchmarks, diagnostics, plots
and paper-specific processing remain in their research folders. Code should be
promoted into ASD when it becomes useful to more than one project without
retaining assumptions from the original research question.
