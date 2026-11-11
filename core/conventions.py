HAMILTONIAN_CONVENTION = "unit_spin"
COEFFICIENT_NAMES = (
    "J1_meV", "J2_meV", "J3_meV", "lambda1_meV", "lambda2_meV", "lambda3_meV", "D_meV",
)


def saved_unit_spin_parameters(parameters):
    parameters = dict(parameters)
    source = parameters.get("hamiltonian_convention", "legacy")
    if source not in (HAMILTONIAN_CONVENTION, "paper", "legacy"):
        raise ValueError(f"unknown saved Hamiltonian convention: {source}")
    if source == "legacy":
        for name in COEFFICIENT_NAMES:
            if name in parameters:
                parameters[name] = -parameters[name]
    parameters["source_hamiltonian_convention"] = source
    parameters["hamiltonian_convention"] = HAMILTONIAN_CONVENTION
    return parameters
