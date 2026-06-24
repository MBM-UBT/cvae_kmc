"""Load dendrite 1d+1 kMC model."""

import numpy as np

import python_kmc.kmc_src.d1_1.kmc_model_1d1 as kmc_model_1d1


def generate_kmc_model(lattice_size) -> kmc_model_1d1.KMCModel_1D1:
    """Generates and returns a KMCModel_1D1 instance with predefined parameters.

    Returns:
        kmc_model_1d1.KMCModel_1D1: 1D+1 KMC model instance with adsorption, desorption,
        and diffusion processes.
    """
    params = {
        "lattice_size": lattice_size,
        "temperature_kelvin": 300,
        "concentration_species_mol_per_liter": 1.0,
        "binding_energy_joule_per_mol": 12000,
        "grid_constant_meter": 1e-10,
        "use_transitionrate_difference": False,
        "seed": 0,
        # [Pre-exponential factor (A), Activation Energy (J/mol), Gibbs shift (J/mol)]
        "adsorption_parameters": np.array([1e7, 11000, 0.0]),
        "desorption_parameters": np.array([1e12, 17000, 0.0]),
        "diffusion_parameters": np.array([1e12, 1700, 0.0]),
        "diffusion_aarhenius": True,
        "allow_arbitrary_diffusion": True,
        "use_transitionrate_difference_adsorption": False,
        "binding_energy_floor_joule_per_mol": 7000,
    }
    my_kmc_model = kmc_model_1d1.KMCModel_1D1(**params)
    return my_kmc_model
