"""Base RateCalculator with common KMC rate calculation logic.

This module provides a base class for rate calculators used by all KMC models,
regardless of dimension. Dimension-specific bonding energy calculations and
rate matrix updates are handled by subclasses.
"""

import os
import sys

import numpy as np
import scipy.constants

# Add core module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", "core"))

GAS_CONSTANT_JOULE_PER_MOL_KELVIN = scipy.constants.R


class BaseRateCalculator:
    """Base class for KMC rate calculators (dimension-agnostic).

    This class provides common initialization and utility methods.
    Subclasses should implement:
    - calculate_transition_rate_constants()
    - update_transition_rate_matrix()
    """

    def __init__(self, model, temperature_kelvin: float, seed: int = 0):
        """Initialize the base RateCalculator.

        Args:
            model: The KMC model instance.
            temperature_kelvin (float): Temperature in Kelvin.
            seed (int): Random seed for reproducibility.
        """
        self.model = model
        self.temperature_kelvin = np.float32(temperature_kelvin)

        # Initialize mutable state arrays
        self.transition_rate_matrix = None
        self.process_sums = None
        self.p_binding_sites = None

        self.set_seed(seed)

    def set_seed(self, seed: int):
        """Set the random seed for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        np.random.seed(seed)

    def reset(self):
        """Reset the RateCalculator to its initial state.

        Subclasses should override this to add dimension-specific logic,
        but should call super().reset() first.
        """
        self.transition_rate_matrix = None
        self.model.total_transition_rate_per_second[0] = 0.0

        # Re-calculate rate constants
        if hasattr(self, "calculate_transition_rate_constants"):
            self.calculate_transition_rate_constants()

    def update_all(self):
        """Update all transition rates and binding energies.

        This is useful when model parameters change significantly.
        """
        self.model.total_transition_rate_per_second[0] = 0.0
        if hasattr(self, "calculate_transition_rate_constants"):
            self.calculate_transition_rate_constants()
        if hasattr(self, "update_transition_rate_matrix"):
            self.update_transition_rate_matrix(indices=-1)
