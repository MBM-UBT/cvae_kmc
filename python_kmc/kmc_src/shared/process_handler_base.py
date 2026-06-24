"""Base ProcessHandler with common KMC process selection logic.

This module provides the shared process selection logic used by all KMC models,
regardless of dimension. Dimension-specific changes (e.g., height vs occupancy updates)
are handled by subclasses.
"""

import numpy as np
from numba import njit


@njit
def select_process_numba(process_column_sums, transition_rate_matrix, random_number):
    """Select a process and site based on a random number.

    This function is dimension-agnostic and works for all KMC models.

    Args:
        process_column_sums (np.ndarray): The column sums of the transition rate matrix.
        transition_rate_matrix (np.ndarray): The transition rate matrix (sites × processes).
        random_number (float): Random number used for selection.

    Returns:
        tuple[int, int]: The selected site and process indices.
    """
    # Step 1: select process (column)
    total = 0.0
    for i in range(process_column_sums.size):
        total += process_column_sums[i]
        if random_number < total:
            chosen_process = i
            total -= process_column_sums[i]
            break

    # Step 2: select site (row) within chosen process
    # Search via the cumulative sum
    col = transition_rate_matrix[:, chosen_process]
    cum = 0.0
    for i in range(col.size):
        cum += col[i]
        if random_number < cum + total:
            return i, chosen_process
    return col.size - 1, chosen_process


class ProcessHandlerBase:
    """Base class for all KMC ProcessHandlers, regardless of dimension.

    This class provides common functionality for process selection.
    Dimension-specific subclasses should override change_config_numba
    to implement their specific state update logic.
    """

    def __init__(self, model, seed: int = 0):
        """Initialize the ProcessHandler with a KMC model.

        Args:
            model: The KMC model instance.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.
        """
        np.random.seed(seed)
        self.model = model

    def set_seed(self, seed: int):
        """Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        np.random.seed(seed)

    def select_process(self) -> tuple[int, int]:
        """Wrapper method to call numba-accelerated process selection.

        Returns:
            tuple[int, int]: The selected (site, process) indices.
        """
        w = self.model.total_transition_rate_per_second[0] * np.random.uniform()

        return select_process_numba(
            self.model.rate_calculator.process_sums,
            self.model.rate_calculator.transition_rate_matrix,
            w,
        )

    def change_config(self, site_index: int, chosen_process: int):
        """Abstract method to change lattice configuration.

        Subclasses must implement this to handle dimension-specific state updates.

        Args:
            site_index (int): The site index to update.
            chosen_process (int): The process index to apply.
        """
        raise NotImplementedError("Subclasses must implement change_config()")
