import time
import tkinter as tk

from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np

from initialise import init_spins
from lattice import build_neighbour_list
from noise import init_quantum_noise
from noise import quantum_noise_amplitudes
from evolve import evolve_quantum_time_series
from evolve import evolve_classical_time_series
from evolve import evolve_deterministic_time_series


def get_parameters():
    params = {}

    root = tk.Tk()
    root.title("Marmalade parameters")

    entries = {}

    defaults = {
        "noise_mode": "classical",
        "initial_state": "aligned",
        "T": "100.0",
        "Lx": "32",
        "Ly": "32",
        "dt": "1e-16",
        "end_time": "1e-12",
        "stride": "10",
        "J": "1e-2",
        "K": "1e-4",
        "lam": "0.01",
    }

    labels = {
        "noise_mode": "Noise mode",
        "initial_state": "Initial state",
        "T": "T",
        "Lx": "Lx",
        "Ly": "Ly",
        "dt": "dt",
        "end_time": "End time",
        "stride": "Stride",
        "J": "J",
        "K": "K",
        "lam": "λ",
    }

    row = 0

    for key, value in defaults.items():
        label = ttk.Label(root, text=labels[key])
        label.grid(row=row, column=0, padx=8, pady=4, sticky="w")

        entry = ttk.Entry(root, width=20)
        entry.insert(0, value)
        entry.grid(row=row, column=1, padx=8, pady=4)

        entries[key] = entry
        row += 1

    def submit():
        params["noise_mode"] = entries["noise_mode"].get()
        params["initial_state"] = entries["initial_state"].get()
        params["T"] = float(entries["T"].get())
        params["Lx"] = int(entries["Lx"].get())
        params["Ly"] = int(entries["Ly"].get())
        params["dt"] = float(entries["dt"].get())
        params["end_time"] = float(entries["end_time"].get())
        params["stride"] = int(entries["stride"].get())
        params["J"] = float(entries["J"].get())
        params["K"] = float(entries["K"].get())
        params["lam"] = float(entries["lam"].get())

        root.withdraw()

        root.update_idletasks()

        root.destroy()

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return params


def plot_magnetisation(t_series, M_series):
    plt.figure(figsize=(7, 4), dpi=150)
    plt.plot(t_series * 1e12, M_series)
    plt.xlabel("Time (ps)")
    plt.ylabel(r"$|\mathbf{M}|$")
    plt.tight_layout()
    plt.show()


params = get_parameters()

noise_mode = params["noise_mode"]
initial_state = params["initial_state"]

T = params["T"]

Lx = params["Lx"]
Ly = params["Ly"]
N = Lx * Ly

dt = params["dt"]
end_time = params["end_time"]
iterations = int(end_time / dt)

stride = params["stride"]

J = params["J"]
K = params["K"]

mu = 9.274e-24
gamma = 1.7609e11
lam = params["lam"]
hbar = 1.054571817e-34
kB = 1.380649e-23
eV = 1.602176634e-19

J *= eV
K *= eV

J_mu = J / mu
K_mu = K / mu

kT = kB * T

Gamma = lam * kT / ((1.0 + lam * lam) * gamma * mu)

dt_q = dt * kT / hbar

q_pref = np.sqrt(2.0 * Gamma * kT / hbar)
c_pref = np.sqrt(2.0 * Gamma / dt)

amp5, amp6 = quantum_noise_amplitudes(dt_q)

S = init_spins(Lx, Ly, mode=initial_state)
neighbour_list, next_neighbour_list = build_neighbour_list(Lx, Ly)

n_samples = (iterations + stride - 1) // stride

t_series = np.empty(n_samples, dtype=np.float64)
M_series = np.empty(n_samples, dtype=np.float64)

print("Running simulation...")

t0 = time.time()

if noise_mode == "quantum":
    z5, v5, z6, v6 = init_quantum_noise(N)
    evolve_quantum_time_series(S, neighbour_list, z5, v5, z6, v6, t_series, M_series, dt, gamma, lam, J_mu, K_mu, q_pref, dt_q, amp5, amp6, iterations, stride)

elif noise_mode == "classical":
    evolve_classical_time_series(S, neighbour_list, t_series, M_series, dt, gamma, lam, J_mu, K_mu, c_pref, iterations, stride)

elif noise_mode == "none":
    evolve_deterministic_time_series(S, neighbour_list, t_series, M_series, dt, gamma, lam, J_mu, K_mu, iterations, stride)

else:
    raise ValueError("noise_mode must be 'quantum', 'classical' or 'none'")

t1 = time.time()

print(f"Simulation time: {t1 - t0:.3f} s")

plot_magnetisation(t_series, M_series)