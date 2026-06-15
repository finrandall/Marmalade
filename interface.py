import tkinter as tk

from tkinter import ttk


def get_parameters():
    params = {}

    root = tk.Tk()
    root.title("Marmalade parameters")

    entries = {}

    defaults = {
        "output_mode": "magnetisation",
        "noise_mode": "classical",
        "model_mode": "fm",
        "initial_state": "fm",
        "T": "100.0",
        "Lx": "32",
        "Ly": "32",
        "dt": "1e-16",
        "end_time": "1e-12",
        "burn_in_time": "1e-13",
        "stride": "10",
        "J1": "1e-2",
        "J2": "0.0",
        "K": "1e-4",
        "h": "0.0",
        "lam": "0.01",
    }

    labels = {
        "output_mode": "Output mode",
        "noise_mode": "Noise mode",
        "model_mode": "Model mode",
        "initial_state": "Initial state",
        "T": "T",
        "Lx": "Lx",
        "Ly": "Ly",
        "dt": "dt",
        "end_time": "End time",
        "burn_in_time": "Burn-in time",
        "stride": "Stride",
        "J1": "J₁",
        "J2": "J₂",
        "K": "K",
        "h": "h",
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
        params["output_mode"] = entries["output_mode"].get()
        params["noise_mode"] = entries["noise_mode"].get()
        params["model_mode"] = entries["model_mode"].get()
        params["initial_state"] = entries["initial_state"].get()
        params["T"] = float(entries["T"].get())
        params["Lx"] = int(entries["Lx"].get())
        params["Ly"] = int(entries["Ly"].get())
        params["dt"] = float(entries["dt"].get())
        params["end_time"] = float(entries["end_time"].get())
        params["burn_in_time"] = float(entries["burn_in_time"].get())
        params["stride"] = int(entries["stride"].get())
        params["J1"] = float(entries["J1"].get())
        params["J2"] = float(entries["J2"].get())
        params["K"] = float(entries["K"].get())
        params["h"] = float(entries["h"].get())
        params["lam"] = float(entries["lam"].get())

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return params