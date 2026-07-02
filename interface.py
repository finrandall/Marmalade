import tkinter as tk
from tkinter import ttk

def get_parameters():
    params: dict[str, object] = {}
    root = tk.Tk()
    root.title("Marmalade parameters")
    root.minsize(420, 300)

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
        "integrator": {
            "Heun": "heun",
            "RKMK2": "rkmk2",
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

    value_to_display = {
        key: {v: k for k, v in mapping.items()}
        for key, mapping in display_to_value.items()
    }

    defaults = {
        "output_mode": "magnetisation",
        "noise_mode": "classical",
        "integrator": "heun",
        "initial_state": "fm",
        "dt": "1e-15",
        "end_time": "1e-11",
        "stride": "1",
        "J1": "1e-2",
        "J2": "0.0",
        "K": "0.0",
        "h": "0.0",
        "T": "30.0",
        "lam": "0.01",
        "Lx": "32",
        "Ly": "32",
        "Lz": "32",
        "pbc_x": True,
        "pbc_y": True,
        "pbc_z": True,
        "burn_in_time": "0.0",
        "spectrum_path": "high_symmetry",
        "show_analytic": True,
    }

    labels = {
        "output_mode": "Output mode",
        "noise_mode": "Noise mode",
        "integrator": "Integrator",
        "initial_state": "Initial state",
        "T": "Temperature",
        "Lx": "Lattice size x",
        "Ly": "Lattice size y",
        "Lz": "Lattice size z",
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

    parameter_groups = {
        "Run options": ["output_mode", "noise_mode", "integrator", "initial_state"],
        "Time constants": ["dt", "end_time", "stride"],
        "Simulation constants": ["J1", "J2", "K", "h", "T", "lam"],
        "Lattice parameters": ["Lx", "Ly", "Lz", "pbc_group"],
    }

    entries: dict[str, object] = {}
    row = 0
    for group_title, group_keys in parameter_groups.items():
        frame = ttk.LabelFrame(root, text=group_title)
        frame.grid(row=row, column=0, columnspan=2, padx=8, pady=6, sticky="ew")
        row += 1
        group_row = 0
        for key in group_keys:
            if key == "pbc_group":
                lbl = ttk.Label(frame, text="Periodic boundaries")
                lbl.grid(row=group_row, column=0, padx=8, pady=4, sticky="w")
                var_x = tk.BooleanVar(value=defaults["pbc_x"])
                var_y = tk.BooleanVar(value=defaults["pbc_y"])
                var_z = tk.BooleanVar(value=defaults["pbc_z"])
                pbc_frame = ttk.Frame(frame)
                chk_x = ttk.Checkbutton(pbc_frame, text="X", variable=var_x)
                chk_y = ttk.Checkbutton(pbc_frame, text="Y", variable=var_y)
                chk_z = ttk.Checkbutton(pbc_frame, text="Z", variable=var_z)
                for chk in (chk_x, chk_y, chk_z):
                    chk.pack(side="left", expand=True, fill="x", padx=4)
                pbc_frame.grid(
                    row=group_row, column=1, columnspan=3, padx=8, pady=4, sticky="ew"
                )
                frame.columnconfigure(1, weight=1)
                frame.columnconfigure(2, weight=1)
                frame.columnconfigure(3, weight=1)
                entries["pbc_x"] = var_x
                entries["pbc_y"] = var_y
                entries["pbc_z"] = var_z
                group_row += 1
                continue
            label = ttk.Label(frame, text=labels[key])
            label.grid(row=group_row, column=0, padx=8, pady=4, sticky="w")
            if key in display_to_value:
                options = list(display_to_value[key].keys())
                combo = ttk.Combobox(frame, values=options, width=18, state="readonly")
                combo.set(value_to_display[key][defaults[key]])
                combo.grid(row=group_row, column=1, padx=8, pady=4, sticky="w")
                entries[key] = combo
            else:
                entry = ttk.Entry(frame, width=20)
                entry.insert(0, str(defaults[key]))
                entry.grid(row=group_row, column=1, padx=8, pady=4, sticky="w")
                entries[key] = entry
            group_row += 1

    def open_spectrum_parameters():
        window = tk.Toplevel(root)
        window.title("Spectrum parameters")
        spectrum_entries: dict[str, object] = {}
        keys = ["burn_in_time", "spectrum_path", "show_analytic"]
        labels_map = {
            "burn_in_time": "Burn-in time",
            "spectrum_path": "Spectrum path",
            "show_analytic": "Show analytic curve",
        }
        row_idx = 0
        for k in keys:
            lbl = ttk.Label(window, text=labels_map[k])
            lbl.grid(row=row_idx, column=0, padx=8, pady=4, sticky="w")
            if k in display_to_value:
                opts = list(display_to_value[k].keys())
                cmb = ttk.Combobox(window, values=opts, width=18, state="readonly")
                cmb.set(value_to_display[k][defaults[k]])
                cmb.grid(row=row_idx, column=1, padx=8, pady=4, sticky="w")
                spectrum_entries[k] = cmb
            else:
                ent = ttk.Entry(window, width=20)
                ent.insert(0, str(defaults[k]))
                ent.grid(row=row_idx, column=1, padx=8, pady=4, sticky="w")
                spectrum_entries[k] = ent
            row_idx += 1

        def save_spectrum_parameters():
            defaults["burn_in_time"] = spectrum_entries["burn_in_time"].get()
            defaults["spectrum_path"] = display_to_value["spectrum_path"][
                spectrum_entries["spectrum_path"].get()
            ]
            defaults["show_analytic"] = display_to_value["show_analytic"][
                spectrum_entries["show_analytic"].get()
            ]
            window.destroy()

        ttk.Button(window, text="Save", command=save_spectrum_parameters).grid(
            row=row_idx, column=0, columnspan=2, padx=8, pady=10
        )

    def submit():
        params["output_mode"] = display_to_value["output_mode"][
            entries["output_mode"].get()
        ]
        params["noise_mode"] = display_to_value["noise_mode"][
            entries["noise_mode"].get()
        ]
        params["integrator"] = display_to_value["integrator"][
            entries["integrator"].get()
        ]
        params["initial_state"] = display_to_display_err_display = lambda: value_to_display; params["initial_state"] = display_to_value["initial_state"][
            entries["initial_state"].get()
        ]
        params["T"] = float(entries["T"].get())
        params["Lx"] = int(entries["Lx"].get())
        params["Ly"] = int(entries["Ly"].get())
        params["Lz"] = int(entries["Lz"].get())
        params["pbc_x"] = bool(entries["pbc_x"].get())
        params["pbc_y"] = bool(entries["pbc_y"].get())
        params["pbc_z"] = bool(entries["pbc_z"].get())
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

    ttk.Button(
        root, text="Spectrum parameters", command=open_spectrum_parameters
    ).grid(row=row, column=0, columnspan=2, padx=8, pady=4)
    row += 1
    ttk.Button(root, text="Run simulation", command=submit).grid(
        row=row, column=0, columnspan=2, padx=8, pady=10
    )

    root.mainloop()
    return params