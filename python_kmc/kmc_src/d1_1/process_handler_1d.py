"""1D+1 ProcessHandler for KMC Simulation"""


import numpy as np
from numba import njit

from python_kmc.kmc_src.shared.process_handler_base import ProcessHandlerBase


class ProcessHandler(ProcessHandlerBase):
    """1D+1 ProcessHandler for KMC Simulation"""

    @staticmethod
    @njit
    def change_config_numba(
        site_index: int,
        chosen_process: int,
        heights: np.ndarray,
        neighbors_array: np.ndarray,
    ):
        """Change the configuration of the lattice.

        Args:
            site_index (int): The site to be updated.
            chosen_process (int): The process to be applied.
            heights (np.ndarray): The heights array.
            neighbors_array (np.ndarray): The neighbor array.
        """

        if chosen_process == 0:  # adsorption
            heights[site_index] += 1  # Set site to occupied
            return 0

        if chosen_process == 1:  # desorption
            heights[site_index] -= 1  # Set site to unoccupied
            return 0

        if chosen_process == 2:  # diffusion to the left
            heights[site_index] -= 1
            heights[neighbors_array[site_index, 0]] += 1
            return 0

        if chosen_process == 3:  # diffusion to the right
            heights[site_index] -= 1
            heights[neighbors_array[site_index, 1]] += 1
            return 0

        return -1  # No valid process selected

    def change_config(self, site_index: int, chosen_process: int):
        """
        Wrapper method to call numba-accelerated change_config_numba.
        """

        # Call the jitted function
        result = ProcessHandler.change_config_numba(
            site_index=site_index,
            chosen_process=chosen_process,
            heights=self.model.lattice.heights,
            neighbors_array=self.model.lattice.neighbors_array,
        )

        if result == -1:
            raise ValueError(
                f"No valid process selected (initial Process ID: {chosen_process})"
            )
