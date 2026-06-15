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
        "model_mode": {
            "Ferromagnetic": "fm",
            "Antiferromagnetic": "afm",
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
    }

    value_to_display = {}

    for key, mapping in display_to_value.items():
        value_to_display[key] = {value: display for display, value in mapping.items()}

    defaults = {
        "output_mode": "spectrum",
        "noise_mode": "quantum",
        "model_mode": "fm",
        "initial_state": "fm",
        "T": "10.0",
        "Lx": "64",
        "Ly": "64",
        "dt": "3e-16",
        "end_time": "6e-12",
        "burn_in_time": "1e-12",
        "spectrum_path": "high_symmetry",
        "show_analytic": "True",
        "stride": "10",
        "J1": "1e-2",
        "J2": "0.0",
        "K": "1e-4",
        "h": "0.0",
        "lam": "0.0001",
    }

    labels = {
        "output_mode": "Output mode",
        "noise_mode": "Noise mode",
        "model_mode": "Model type",
        "initial_state": "Initial state",
        "T": "Temperature",
        "Lx": "Lattice size x",
        "Ly": "Lattice size y",
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
    hidden_keys = {"burn_in_time", "spectrum_path", "show_analytic"}

    for key, value in defaults.items():
        if key in hidden_keys:
            continue

        label = ttk.Label(root, text=labels[key])
        label.grid(row=row, column=0, padx=8, pady=4, sticky="w")

        if key in display_to_value:
            entry = ttk.Combobox(root, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
            entry.set(value_to_display[key][value])
        else:
            entry = ttk.Entry(root, width=20)
            entry.insert(0, value)

        entry.grid(row=row, column=1, padx=8, pady=4)

        entries[key] = entry
        row += 1

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
                entry.set(value_to_display[key][display_to_value[key].get(str(defaults[key]), defaults[key])])
            else:
                entry = ttk.Entry(window, width=20)
                entry.insert(0, str(defaults[key]))

            entry.grid(row=spectrum_row, column=1, padx=8, pady=4)
            spectrum_entries[key] = entry
            spectrum_row += 1

        def save_spectrum_parameters():
            defaults["burn_in_time"] = spectrum_entries["burn_in_time"].get()
            defaults["spectrum_path"] = display_to_value["spectrum_path"][spectrum_entries["spectrum_path"].get()]
            defaults["show_analytic"] = spectrum_entries["show_analytic"].get()

            window.destroy()

        button = ttk.Button(window, text="Save", command=save_spectrum_parameters)
        button.grid(row=spectrum_row, column=0, columnspan=2, padx=8, pady=10)

    def submit():
        params["output_mode"] = display_to_value["output_mode"][entries["output_mode"].get()]
        params["noise_mode"] = display_to_value["noise_mode"][entries["noise_mode"].get()]
        params["model_mode"] = display_to_value["model_mode"][entries["model_mode"].get()]
        params["initial_state"] = display_to_value["initial_state"][entries["initial_state"].get()]
        params["T"] = float(entries["T"].get())
        params["Lx"] = int(entries["Lx"].get())
        params["Ly"] = int(entries["Ly"].get())
        params["dt"] = float(entries["dt"].get())
        params["end_time"] = float(entries["end_time"].get())
        params["burn_in_time"] = float(defaults["burn_in_time"])
        params["spectrum_path"] = defaults["spectrum_path"]
        params["show_analytic"] = display_to_value["show_analytic"][defaults["show_analytic"]]
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