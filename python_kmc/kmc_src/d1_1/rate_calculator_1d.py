"""Rate calculator for the 1D+1 KMC model.

Calculates reaction rates based on model parameters and lattice configuration.
Takes care of the Transition Rate Matrix (TRM) updates.
"""

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import scipy.constants
from numba import njit

from python_kmc.kmc_src.d1_1.lattice_1d import get_numbers_of_occupied_neighbors
from python_kmc.kmc_src.shared.rate_calculator_base import BaseRateCalculator

GAS_CONSTANT_JOULE_PER_MOL_KELVIN = scipy.constants.R
FARADAY_CONSTANT_AS_PER_MOL = scipy.constants.physical_constants['Faraday constant'][0]

@njit
def compute_p_binding_sites_numba(
    indices: np.array,
    binding_energy_joule_per_mol: np.array,
    temperature_kelvin: float,
    heights: np.array,
    neighbors_array: np.array,
    gas_constant_joule_per_mol_kelvin: float,  # this needs to be passed for numba compatibility
    adsorption: bool = False,
    diffusion_direction: int = -1,
    binding_energy_floor_joule_per_mol: float = 0.0,
):
    """Compute the bond energy for each site in the lattice.

    Args:
        indices (np.array): Indices of the sites to compute.
        binding_energy_joule_per_mol (np.array): Bond energy matrix for the species.
        temperature_kelvin (float): Temperature in Kelvin.
        heights (np.array): Heights of the lattice sites.
        neighbors_array (np.array): Neighbor array.
        gas_constant_joule_per_mol_kelvin (float): Gas constant in J/(mol*K).
        adsorption (bool, optional): If True, consider adsorption. Defaults to False.
        diffusion_direction (int, optional): Direction of diffusion (-1 if not diffusion). Defaults to -1.
        binding_energy_floor_joule_per_mol (float, optional): Floor for binding energy in J/mol. Defaults to 0.0.
    Returns:
        np.array: Array of bond energies for each site.
    """
    # get direct and diagonal neighbors occupation
    neighbors_direct, neighbors_diagonal, bottom_occupied = (
        get_numbers_of_occupied_neighbors(
            indices=indices,
            heights=heights,
            neighbors=neighbors_array,
            adsorption=adsorption,
            diffusion_direction=diffusion_direction,
        )
    )

    # Calculate the total bonding energy from direct and diagonal neighbors
    direct_bonding_energies = neighbors_direct * binding_energy_joule_per_mol
    diagonal_bonding_energies = (
        neighbors_diagonal * binding_energy_joule_per_mol * np.sqrt(0.5)
    )
    bonding_floor = binding_energy_floor_joule_per_mol * (
        bottom_occupied == 0
    )  # Apply floor only if there is no particle below
    total_bonding_energies = (
        direct_bonding_energies + diagonal_bonding_energies + bonding_floor
    )

    # Calculate the bond energy for each site
    p_binding = np.exp(
        -(total_bonding_energies)
        / (gas_constant_joule_per_mol_kelvin * temperature_kelvin)
    )

    return p_binding


@njit
def update_transition_rate_matrix_numba(
    transition_rate_matrix: np.array,
    p_binding_sites: np.array,
    concentration_species_mol_per_liter: np.array,
    heights: np.array,
    update_indices: np.array,
    neighbors_array: np.array,
    base_rates: np.array,
    total_transition_rate_per_second: np.array,
    process_sums: np.array,
    temperature_kelvin: float,
    binding_energy_joule_per_mol: np.array,
    use_transitionrate_difference: bool = False,
    binding_energy_floor_joule_per_mol: float = 0.0,
    allow_arbitrary_diffusion: bool = False,
    use_transitionrate_difference_adsorption: bool = True,
):
    """Update the transition rate matrix after a process has taken place.

    Using numba for performance.

    Args:
        transition_rate_matrix (np.array): Process matrix to update.
        p_binding_sites (np.array): Bond energy array.
        concentration_species_mol_per_liter (np.array): Concentration of species.
        heights (np.array): Heights of the lattice sites.
        update_indices (np.array): Indices of the sites to update.
        neighbors_array (np.array): Neighbor array.
        possible_reactions_one_species (np.array): First reaction array.
        possible_reactions_diffusion (np.array): Diffusion reaction array.
        total_transition_rate_per_second (np.array): Sum of all transition rates.
        process_sums (np.array): Process sums for each process.
        temperature_kelvin (float): Temperature in Kelvin.
        binding_energy_joule_per_mol (np.array): Bond energy matrix.
        use_transitionrate_difference (bool, optional): If True, consider the change in bonding energy for reactions. Defaults to False.
        binding_energy_floor_joule_per_mol (float, optional): Floor for binding energy in J/mol. Defaults to 0.0.
        allow_arbitrary_diffusion (bool, optional): Whether to allow diffusion regardless of height difference. Defaults to False.
        use_transitionrate_difference_adsorption (bool, optional): Whether to use transition rate difference for adsorption. Defaults to True.
    Returns:
        None
    """
    # Save old rates before updating:
    # old_process_sums = np.sum(transition_rate_matrix[update_indices, :], axis=0)

    # Iterate over all the columns (processes) in the transition_rate_matrix and update the rates
    current_column = 0
    # Fill transition_rate_matrix with reaction rates for only one reactant
    for i, rate in enumerate(base_rates):
        if i == 0:  # adsorption
            transition_rate_matrix[update_indices, current_column] = (
                rate * concentration_species_mol_per_liter
            )
            # If true, calculate the potential bonding energy change if this reaction occurs
            if (
                use_transitionrate_difference
                and use_transitionrate_difference_adsorption
            ):
                p_binding_pot = compute_p_binding_sites_numba(
                    indices=update_indices,
                    binding_energy_joule_per_mol=binding_energy_joule_per_mol,
                    temperature_kelvin=temperature_kelvin,
                    heights=heights,
                    neighbors_array=neighbors_array,
                    gas_constant_joule_per_mol_kelvin=GAS_CONSTANT_JOULE_PER_MOL_KELVIN,
                    adsorption=True,
                    binding_energy_floor_joule_per_mol=binding_energy_floor_joule_per_mol,
                )
                transition_rate_matrix[update_indices, current_column] /= p_binding_pot

        if i == 1:  # desorption
            active_sites = heights[update_indices] > 0
            transition_rate_matrix[update_indices[~active_sites], current_column] = 0.0
            transition_rate_matrix[update_indices[active_sites], current_column] = (
                rate * p_binding_sites[update_indices[active_sites]]
            )

        if i == 2:  # diffusion
            for dir in range(2):  # 2 directions: left and right
                neighbor_heights = heights[neighbors_array[update_indices, dir]]

                # Reset all rates to zero first
                transition_rate_matrix[update_indices, current_column + dir] = 0.0

                if allow_arbitrary_diffusion:
                    # --- NEW SIMPLIFIED LOGIC ---
                    # 1. Allow diffusion to ANY neighbor height.
                    # 2. No penalty for diagonal/vertical moves (use full 'rate').

                    # The only condition is that the source stack must have a particle to move.
                    active_indices = heights[update_indices] > 0

                    transition_rate_matrix[
                        update_indices[active_indices],
                        current_column + dir,
                    ] = (
                        rate  # No division by 2
                        * p_binding_sites[update_indices[active_indices]]
                    )

                else:
                    # --- ORIGINAL RESTRICTED LOGIC (Nearest Neighbors Only) ---
                    # Even here, we removed the /2 penalty as requested.

                    # 1. Direct diffusion (Flat)
                    active_direct_indices = (
                        heights[update_indices] - 1 == neighbor_heights
                    ) & (heights[update_indices] > 0)

                    transition_rate_matrix[
                        update_indices[active_direct_indices],
                        current_column + dir,
                    ] = (
                        rate * p_binding_sites[update_indices[active_direct_indices]]
                    )

                    # 2. Diagonal Upper (Step Up)
                    active_upper_indices = (
                        heights[update_indices] == neighbor_heights
                    ) & (heights[update_indices] > 0)

                    transition_rate_matrix[
                        update_indices[active_upper_indices],
                        current_column + dir,
                    ] = (
                        rate * p_binding_sites[update_indices[active_upper_indices]]
                    ) / 2

                    # 3. Diagonal Lower (Step Down)
                    active_lower_indices = (
                        heights[update_indices] == neighbor_heights + 2
                    ) & (heights[update_indices] > 1)

                    transition_rate_matrix[
                        update_indices[active_lower_indices],
                        current_column + dir,
                    ] = (
                        rate * p_binding_sites[update_indices[active_lower_indices]]
                    ) / 2

                # --- Energy Correction ---
                if use_transitionrate_difference:
                    p_binding_pot = compute_p_binding_sites_numba(
                        indices=neighbors_array[update_indices, dir],
                        binding_energy_joule_per_mol=binding_energy_joule_per_mol,
                        temperature_kelvin=temperature_kelvin,
                        heights=heights,
                        neighbors_array=neighbors_array,
                        gas_constant_joule_per_mol_kelvin=GAS_CONSTANT_JOULE_PER_MOL_KELVIN,
                        adsorption=False,
                        diffusion_direction=dir,
                        binding_energy_floor_joule_per_mol=binding_energy_floor_joule_per_mol,
                    )
                    transition_rate_matrix[
                        update_indices, current_column + dir
                    ] /= p_binding_pot

        # Move to the next process column
        current_column += 1

    # New rates after updating:
    process_sums[:] = np.sum(transition_rate_matrix, axis=0)

    # Update the total_transition_rate and process_sums
    total_transition_rate_per_second[0] = np.sum(process_sums)


class RateCalculator(BaseRateCalculator):
    """Rate Calculator Class for 1D+1 KMC Simulation"""

    def __init__(
        self,
        model,
        temperature_kelvin: float,
        concentration_species_mol_per_liter: np.array,
        binding_energy_joule_per_mol: np.array,
        adsorption_parameters: np.array,
        desorption_parameters: np.array,
        diffusion_parameters: np.array,
        grid_constant_meter: float,
        use_transitionrate_difference: bool = False,
        binding_energy_floor_joule_per_mol: float = 0.0,
        diffusion_aarhenius: bool = False,
        allow_arbitrary_diffusion: bool = False,
        use_transitionrate_difference_adsorption: bool = True,
        seed: int = 0,
    ):
        """Initialize the Rate Calculator for 1D+1 KMC Simulation.

        Args:
            model: KMC model instance.
            temperature_kelvin (float): [K] Temperature.
            concentration_species_mol_per_liter (np.array): [mol/l] Concentration of species.
            binding_energy_joule_per_mol (np.array): [J/mol] Bond energy for the species
            adsorption_parameters (np.array): Parameters for adsorption process.
            desorption_parameters (np.array): Parameters for desorption process.
            diffusion_parameters (np.array): Parameters for diffusion process.
            grid_constant_meter (float): [m] Distance between two lattice sites.
            use_transitionrate_difference (bool, optional): If True, consider bonding energy changes. Defaults to False.
            binding_energy_floor_joule_per_mol (float, optional): Floor for binding energy. Defaults to 0.0.
            diffusion_aarhenius (bool, optional): Whether to use Arrhenius diffusion. Defaults to False.
            allow_arbitrary_diffusion (bool, optional): Allow diffusion regardless of height. Defaults to False.
            use_transitionrate_difference_adsorption (bool, optional): Use energy difference for adsorption. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        # Call parent class initialization
        super().__init__(model=model, temperature_kelvin=temperature_kelvin, seed=seed)

        # Store dimension-specific parameters
        self.concentration_species_mol_per_liter = np.float32(
            concentration_species_mol_per_liter
        )
        self.binding_energy_joule_per_mol = np.float32(binding_energy_joule_per_mol)
        self.grid_constant_meter = np.float32(grid_constant_meter)

        self.adsorption_parameters = np.float32(adsorption_parameters)
        self.desorption_parameters = np.float32(desorption_parameters)
        self.diffusion_parameters = np.float32(diffusion_parameters)

        self.use_transitionrate_difference = use_transitionrate_difference
        self.binding_energy_floor_joule_per_mol = binding_energy_floor_joule_per_mol
        self.diffusion_aarhenius = diffusion_aarhenius
        self.allow_arbitrary_diffusion = allow_arbitrary_diffusion
        self.use_transitionrate_difference_adsorption = (
            use_transitionrate_difference_adsorption
        )

        # Calculate the rate constants for all reactions
        self.calculate_transition_rate_constants()

        # Initialize the p_binding_sites array
        self.p_binding_sites = np.ones(
            self.model.lattice.size,
            dtype=np.float32,
        )

        # Calculate p_binding_sites for all sites in the lattice
        self.p_binding_sites = compute_p_binding_sites_numba(
            indices=np.arange(self.model.lattice.size),
            binding_energy_joule_per_mol=self.binding_energy_joule_per_mol,
            temperature_kelvin=self.temperature_kelvin,
            heights=self.model.lattice.heights,
            neighbors_array=self.model.lattice.neighbors_array,
            gas_constant_joule_per_mol_kelvin=GAS_CONSTANT_JOULE_PER_MOL_KELVIN,
            adsorption=False,
            binding_energy_floor_joule_per_mol=self.binding_energy_floor_joule_per_mol,
        )

        # Initialize the transition rate matrix (TRM)
        self.update_transition_rate_matrix(indices=-1)  # -1 means update all sites

    def calculate_transition_rate_constants(self):
        """Calculate the rate constants k for all possible reactions."""

        self.base_rates = np.zeros(3, dtype=np.float32)

        # --- Adsorption ---
        gibbs_energy_joule_per_mol = np.where(
            self.adsorption_parameters[2] < 0,
            0,
            self.adsorption_parameters[2],
        )

        total_delta_energy_joule_per_mol = (
            gibbs_energy_joule_per_mol + self.adsorption_parameters[1]
        )  # dE_Gibbs_Jpmol + dE_Activation_Jpmol

        self.base_rates[0] = self.adsorption_parameters[0] * np.exp(
            -total_delta_energy_joule_per_mol
            / (GAS_CONSTANT_JOULE_PER_MOL_KELVIN * self.temperature_kelvin)
        )

        # --- Desorption ---
        gibbs_energy_joule_per_mol = np.where(
            self.desorption_parameters[2] < 0,
            0,
            self.desorption_parameters[2],
        )

        total_delta_energy_joule_per_mol = (
            gibbs_energy_joule_per_mol + self.desorption_parameters[1]
        )  # dE_Gibbs_Jpmol + dE_Activation_Jpmol

        self.base_rates[1] = self.desorption_parameters[0] * np.exp(
            -total_delta_energy_joule_per_mol
            / (GAS_CONSTANT_JOULE_PER_MOL_KELVIN * self.temperature_kelvin)
        )

        # --- Diffusion ---
        if self.diffusion_aarhenius:
            gibbs_energy_joule_per_mol = np.where(
                self.diffusion_parameters[2] < 0,
                0,
                self.diffusion_parameters[2],
            )

            total_delta_energy_joule_per_mol = (
                gibbs_energy_joule_per_mol + self.diffusion_parameters[1]
            )  # dE_Gibbs_Jpmol + dE_Activation_Jpmol

            self.base_rates[2] = self.diffusion_parameters[0] * np.exp(
                -total_delta_energy_joule_per_mol
                / (GAS_CONSTANT_JOULE_PER_MOL_KELVIN * self.temperature_kelvin)
            )
        else:
            self.base_rates[2] = (
                self.diffusion_parameters[0]
                / (2 * self.grid_constant_meter**2)
                * np.exp(
                    -self.diffusion_parameters[1]
                    / (GAS_CONSTANT_JOULE_PER_MOL_KELVIN * self.temperature_kelvin)
                )
            )

    def update_transition_rate_matrix(self, indices: int = -1, process: int = 0):
        """Wrapper for the numba function to update the transition rate matrix.

        Args:
            indices (int, optional): The indices to update. Defaults to -1. -1 updates all sites.
            process (int, optional): The process to update. Defaults to 0.
        """
        if indices == -1:
            # Calculate the number of columns in the transition_rate_matrix.
            # number of possible reactions: adsorption, desorption, diffusion left, diffusion right
            num_columns = 4

            # Create empty Matrix. Each Row is for one spot in the lattice.
            # Each Column is for one possible process.
            self.transition_rate_matrix = np.zeros(
                (
                    self.model.lattice.size,
                    num_columns,
                ),
                dtype=np.float32,
            )

            # Initialize the process sums array. The sum of each row in the TRM.
            self.process_sums = np.zeros(num_columns, dtype=np.float32)
            # Update all indices
            update_indices = np.arange(self.model.lattice.size)
        else:  # Update only the specified index
            nb_indices = self.model.lattice.neighbors_array[indices, :].reshape(
                -1
            )  # Get the indices of the neighbors
            # check if indices is array or single int
            if isinstance(indices, np.ndarray):
                update_indices = np.concatenate(
                    (indices, nb_indices)
                )  # All indices to update if there is no diffusion
            else:
                update_indices = np.concatenate(
                    (np.array([indices]), nb_indices)
                )  # All indices to update if there is no diffusion

            # If the process is diffusion update the neighbors of the neighbors also.
            if process >= 2:  # diffusion
                nb_nb_indices = self.model.lattice.neighbors_array[
                    nb_indices, :
                ]  # Get the indices of the neighbors of the neighbors
                update_indices = np.concatenate(
                    (update_indices, nb_nb_indices.flatten()), axis=0
                )

        update_indices = np.unique(update_indices)  # Remove duplicates
        # Calculate the p_binding_sites for all the updated indices
        self.p_binding_sites[update_indices] = compute_p_binding_sites_numba(
            indices=update_indices,
            binding_energy_joule_per_mol=self.binding_energy_joule_per_mol,
            temperature_kelvin=self.temperature_kelvin,
            heights=self.model.lattice.heights,
            neighbors_array=self.model.lattice.neighbors_array,
            gas_constant_joule_per_mol_kelvin=GAS_CONSTANT_JOULE_PER_MOL_KELVIN,
            adsorption=False,
            binding_energy_floor_joule_per_mol=self.binding_energy_floor_joule_per_mol,
        )

        # call the numba function to update the transition rate matrix
        update_transition_rate_matrix_numba(
            transition_rate_matrix=self.transition_rate_matrix,
            p_binding_sites=self.p_binding_sites,
            concentration_species_mol_per_liter=self.concentration_species_mol_per_liter,
            heights=self.model.lattice.heights,
            update_indices=update_indices,
            neighbors_array=self.model.lattice.neighbors_array,
            base_rates=self.base_rates,
            total_transition_rate_per_second=self.model.total_transition_rate_per_second,
            process_sums=self.process_sums,
            temperature_kelvin=self.temperature_kelvin,
            binding_energy_joule_per_mol=self.binding_energy_joule_per_mol,
            use_transitionrate_difference=self.use_transitionrate_difference,
            binding_energy_floor_joule_per_mol=self.binding_energy_floor_joule_per_mol,
            allow_arbitrary_diffusion=self.allow_arbitrary_diffusion,
            use_transitionrate_difference_adsorption=self.use_transitionrate_difference_adsorption,
        )

    def reset(self):
        """Reset the RateCalculator to its initial state."""
        # Call parent reset
        super().reset()

        # Create p_binding_sites array
        self.p_binding_sites = np.ones(
            self.model.lattice.size,
            dtype=np.float32,
        )
        # Calculate the bond energy for each site in the lattice
        self.p_binding_sites = compute_p_binding_sites_numba(
            indices=np.arange(self.model.lattice.size, dtype=np.int32),
            binding_energy_joule_per_mol=self.binding_energy_joule_per_mol,
            temperature_kelvin=self.temperature_kelvin,
            heights=self.model.lattice.heights,
            neighbors_array=self.model.lattice.neighbors_array,
            gas_constant_joule_per_mol_kelvin=GAS_CONSTANT_JOULE_PER_MOL_KELVIN,
            adsorption=False,
            binding_energy_floor_joule_per_mol=self.binding_energy_floor_joule_per_mol,
        )
        self.update_transition_rate_matrix(
            indices=-1
        )  # Update the transition rate matrix for all sites
