import tkinter as tk

from tkinter import ttk


def get_parameters():
    params = {}

    root = tk.Tk()
    root.title("Marmalade parameters")

    entries = {}

    display_to_value = {
        "output_mode": {
            "Magnetisation": "magnetisation",
            "Spectrum": "spectrum",
        },
        "noise_mode": {
            "Classical": "classical",
            "Quantum": "quantum",
            "None": "none",
        },
        "initial_state": {
            "Ferromagnetic": "fm",
            "Antiferromagnetic": "afm",
            "Random": "random",
        },
        "spectrum_path": {
            "Kₓ": "kx",
            "Γ-X-M-Γ": "high_symmetry",
        },
        "show_analytic": {
            "True": True,
            "False": False,
        },
        "pbc_x": {
            "True": True,
            "False": False,
        },
        "pbc_y": {
            "True": True,
            "False": False,
        },
    }

    value_to_display = {}

    for key, mapping in display_to_value.items():
        value_to_display[key] = {value: display for display, value in mapping.items()}

    defaults = {
        "output_mode": "spectrum",
        "noise_mode": "quantum",
        "initial_state": "afm",
        "dt": "3e-16",
        "end_time": "2e-11",
        "stride": "10",
        "J1": "-1e-2",
        "J2": "1e-3",
        "K": "1e-4",
        "h": "1e-3",
        "T": "10.0",
        "lam": "0.0001",
        "Lx": "128",
        "Ly": "128",
        "pbc_x": True,
        "pbc_y": True,
        "burn_in_time": "4e-12",
        "spectrum_path": "high_symmetry",
        "show_analytic": True,
    }

    labels = {
        "output_mode": "Output mode",
        "noise_mode": "Noise mode",
        "initial_state": "Initial state",
        "T": "Temperature",
        "Lx": "Lattice size x",
        "Ly": "Lattice size y",
        "pbc_x": "Periodic boundary x",
        "pbc_y": "Periodic boundary y",
        "dt": "Time step",
        "end_time": "End time",
        "burn_in_time": "Burn-in time",
        "spectrum_path": "Spectrum path",
        "show_analytic": "Show analytic curve",
        "stride": "Stride",
        "J1": "Exchange interaction J₁",
        "J2": "Exchange interaction J₂",
        "K": "Anisotropy K",
        "h": "Field h",
        "lam": "Damping λ",
    }

    row = 0

    parameter_groups = {
        "Run options": ["output_mode", "noise_mode", "initial_state"],
        "Time constants": ["dt", "end_time", "stride"],
        "Simulation constants": ["J1", "J2", "K", "h", "T", "lam"],
        "Lattice parameters": ["Lx", "Ly", "pbc_x", "pbc_y"],
    }

    for group_title, group_keys in parameter_groups.items():
        frame = ttk.LabelFrame(root, text=group_title)
        frame.grid(row=row, column=0, columnspan=2, padx=8, pady=6, sticky="ew")
        row += 1

        group_row = 0

        for key in group_keys:
            label = ttk.Label(frame, text=labels[key])
            label.grid(row=group_row, column=0, padx=8, pady=4, sticky="w")

            if key in display_to_value:
                entry = ttk.Combobox(frame, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
                entry.set(value_to_display[key][defaults[key]])
            else:
                entry = ttk.Entry(frame, width=20)
                entry.insert(0, defaults[key])

            entry.grid(row=group_row, column=1, padx=8, pady=4)

            entries[key] = entry
            group_row += 1

    def open_spectrum_parameters():
        window = tk.Toplevel(root)
        window.title("Spectrum parameters")

        spectrum_entries = {}

        spectrum_keys = ["burn_in_time", "spectrum_path", "show_analytic"]

        spectrum_labels = {
            "burn_in_time": "Burn-in time",
            "spectrum_path": "Spectrum path",
            "show_analytic": "Show analytic curve",
        }

        spectrum_row = 0

        for key in spectrum_keys:
            label = ttk.Label(window, text=spectrum_labels[key])
            label.grid(row=spectrum_row, column=0, padx=8, pady=4, sticky="w")

            if key in display_to_value:
                entry = ttk.Combobox(window, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
                entry.set(value_to_display[key][defaults[key]])
            else:
                entry = ttk.Entry(window, width=20)
                entry.insert(0, str(defaults[key]))

            entry.grid(row=spectrum_row, column=1, padx=8, pady=4)
            spectrum_entries[key] = entry
            spectrum_row += 1

        def save_spectrum_parameters():
            defaults["burn_in_time"] = spectrum_entries["burn_in_time"].get()
            defaults["spectrum_path"] = display_to_value["spectrum_path"][spectrum_entries["spectrum_path"].get()]
            defaults["show_analytic"] = display_to_value["show_analytic"][spectrum_entries["show_analytic"].get()]

            window.destroy()

        button = ttk.Button(window, text="Save", command=save_spectrum_parameters)
        button.grid(row=spectrum_row, column=0, columnspan=2, padx=8, pady=10)

    def submit():
        params["output_mode"] = display_to_value["output_mode"][entries["output_mode"].get()]
        params["noise_mode"] = display_to_value["noise_mode"][entries["noise_mode"].get()]
        params["initial_state"] = display_to_value["initial_state"][entries["initial_state"].get()]
        params["T"] = float(entries["T"].get())
        params["Lx"] = int(entries["Lx"].get())
        params["Ly"] = int(entries["Ly"].get())
        params["pbc_x"] = display_to_value["pbc_x"][entries["pbc_x"].get()]
        params["pbc_y"] = display_to_value["pbc_y"][entries["pbc_y"].get()]
        params["dt"] = float(entries["dt"].get())
        params["end_time"] = float(entries["end_time"].get())
        params["burn_in_time"] = float(defaults["burn_in_time"])
        params["spectrum_path"] = defaults["spectrum_path"]
        params["show_analytic"] = defaults["show_analytic"]
        params["stride"] = int(entries["stride"].get())
        params["J1"] = float(entries["J1"].get())
        params["J2"] = float(entries["J2"].get())
        params["K"] = float(entries["K"].get())
        params["h"] = float(entries["h"].get())
        params["lam"] = float(entries["lam"].get())

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    spectrum_button = ttk.Button(root, text="Spectrum parameters", command=open_spectrum_parameters)
    spectrum_button.grid(row=row, column=0, columnspan=2, padx=8, pady=4)
    row += 1

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return params