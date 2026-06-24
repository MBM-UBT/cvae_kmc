"""Base Lattice class for all KMC implementations.

This module provides a common interface for lattices across different dimensions.
Dimension-specific details (neighbor generation, state representation) are handled
by subclasses.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseLattice(ABC):
    """Abstract base class for all KMC lattices.

    Defines the common interface for lattices regardless of dimension.
    Subclasses must implement dimension-specific neighbor generation
    and state initialization.
    """

    def __init__(self, model):
        """Initialize the base lattice.

        Args:
            model: The KMC model instance.
        """
        self.model = model
        self.size = None
        self.neighbors_array = None

    @abstractmethod
    def reset_lattice(self, **kwargs):
        """Reset lattice to empty state.

        Dimension-specific implementations should clear the lattice state.

        Args:
            **kwargs: Dimension-specific parameters (e.g., lattice_size, sites_x, sites_y)
        """
        pass

    @abstractmethod
    def get_state(self):
        """Get the current lattice state.

        Returns:
            np.array: The state array (heights for 1D, occupancy for 2D, etc.)
        """
        pass

    @abstractmethod
    def set_state(self, state: np.array):
        """Set the lattice state from an array.

        Args:
            state (np.array): The state array to set.

        Raises:
            ValueError: If state size doesn't match lattice size.
        """
        pass

    def validate_size(self, expected_size: int, actual_size: int):
        """Validate that a state size matches the lattice size.

        Args:
            expected_size (int): The expected size.
            actual_size (int): The actual size to validate.

        Raises:
            ValueError: If sizes don't match.
        """
        if actual_size != expected_size:
            raise ValueError(
                f"Lattice size mismatch: expected {expected_size}, got {actual_size}"
            )
