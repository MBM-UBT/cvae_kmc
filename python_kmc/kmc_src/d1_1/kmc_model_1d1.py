"""kmc_model for 1d+1 KMC simulations."""

import numpy as np

import python_kmc.kmc_src.d1_1.lattice_1d as lattice_1d
import python_kmc.kmc_src.d1_1.process_handler_1d as process_handler_1d
import python_kmc.kmc_src.d1_1.rate_calculator_1d as rate_calculator_1d
from python_kmc.kmc_src.shared.kmc_model_base import BaseKMCModel


class KMCModel_1D1(BaseKMCModel):
    """1D+1 KMC Model with with peridoc boundary conditions and binding energy."""

    def __init__(
        self,
        lattice_size: int,
        temperature_kelvin: float,
        concentration_species_mol_per_liter: float,
        binding_energy_joule_per_mol: float,
        grid_constant_meter: float,
        adsorption_parameters: np.array,
        desorption_parameters: np.array,
        diffusion_parameters: np.array,
        binding_energy_floor_joule_per_mol: float = 0.0,
        use_transitionrate_difference: bool = False,
        diffusion_aarhenius: bool = False,
        allow_arbitrary_diffusion: bool = False,
        use_transitionrate_difference_adsorption: bool = True,
        seed: int = 0,
    ):
        """Initialze 1D+1 KMC Model

        Args:
            lattice_size (int): Number of sites in the lattice
            temperature_kelvin (float): Temperature in Kelvin
            concentration_species_mol_per_liter (float): Concentration of species in mol/L
            binding_energy_joule_per_mol (float): Binding energy in J/mol
            grid_constant_meter (float): Grid constant in meters
            binding_energy_floor_joule_per_mol (float, optional): Floor for binding energy in J/mol. Defaults to 0.0.
            adsorption_parameters (np.array): Parameters for adsorption process (pre-exponential, activation energy, gibbs energy)
            desorption_parameters (np.array): Parameters for desorption process (pre-exponential, activation energy, gibbs energy)
            diffusion_parameters (np.array): Parameters for diffusion process (pre-exponential, activation energy, gibbs energy)
            use_transitionrate_difference (bool, optional): Whether to use
            transition rate difference. Defaults to False.
            diffusion_aarhenius (bool, optional): Whether to use Arrhenius diffusion. Defaults to False.
            allow_arbitrary_diffusion (bool, optional): Whether to allow arbitrary diffusion (i.e., no binding energy constraints). Defaults to False.
            use_transitionrate_difference_adsorption (bool, optional): Whether to use transition rate difference for adsorption. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        # Initialize the base class
        super().__init__(seed=seed)

        # set up rate calculator parameters as a dictionary for easier passing
        rate_caclulator_params = {
            "model": self,
            "temperature_kelvin": temperature_kelvin,
            "concentration_species_mol_per_liter": concentration_species_mol_per_liter,
            "binding_energy_joule_per_mol": binding_energy_joule_per_mol,
            "grid_constant_meter": grid_constant_meter,
            "use_transitionrate_difference": use_transitionrate_difference,
            "seed": seed,
            "adsorption_parameters": adsorption_parameters,
            "desorption_parameters": desorption_parameters,
            "diffusion_parameters": diffusion_parameters,
            "binding_energy_floor_joule_per_mol": binding_energy_floor_joule_per_mol,
            "diffusion_aarhenius": diffusion_aarhenius,
            "allow_arbitrary_diffusion": allow_arbitrary_diffusion,
            "use_transitionrate_difference_adsorption": use_transitionrate_difference_adsorption,
        }

        # Set up kMC components based on dimension
        self.lattice = lattice_1d.Lattice(model=self, lattice_size=lattice_size)
        self.process_handler = process_handler_1d.ProcessHandler(model=self, seed=seed)
        self.rate_calculator = rate_calculator_1d.RateCalculator(
            **rate_caclulator_params
        )

    def reset(self, lattice_size: int = None, heights: np.array = None):
        """Reset the model to its initial state.

        Args:
            lattice_size (int, optional): New size for the lattice. Defaults to None.
            heights (np.array, optional): New heights configuration. Defaults to None.
        """
        self.time_passed_seconds = 0  # reset passed time
        self.loop_counter = 0  # reset loop counter

        if (
            lattice_size is None and heights is None
        ):  # If no new shape is provided and no new lattice, keep the current shape
            self.lattice.reset_lattice()
        elif lattice_size is not None:  # If new shape is provided, reset to new shape
            self.lattice.reset_lattice(lattice_size=lattice_size)
        elif heights is not None:  # If new lattice is provided, set the lattice
            self.lattice.set_lattice_heights(heights=heights)

        # Reset the total reaction rate
        self.total_transition_rate_per_second = np.array([0], dtype=np.float32)

        self.rate_calculator.reset()
