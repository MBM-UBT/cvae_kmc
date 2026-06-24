"""1D+1 Lattice for KMC simulations with height-based representation."""


import numpy as np
from numba import njit

from python_kmc.kmc_src.shared.lattice_base import BaseLattice
from python_kmc.kmc_src.shared.neighbor_generation import generate_neighbors_1d


@njit
def get_numbers_of_occupied_neighbors(
    indices: np.array,
    heights: np.array,
    neighbors: np.array,
    adsorption: bool = False,
    diffusion_direction: int = -1,
) -> tuple[np.array, np.array]:
    """Get the number of direct and diagonal occupied neighbors

    Args:
        indices (np.array): indices to check for occupied neighbors
        heights (np.array): heights of the lattice sites
        neighbors (np.array): neighbor array
        adsorption (bool, optional): If True, consider adsorption. Defaults to False.
        diffusion_direction (int, optional): Direction of diffusion (-1 if not diffusion). Defaults to -1.

    Returns:
        tuple[np.array, np.array]: _description_
    """
    # 0 is left neighbor, 1 is right neighbor
    left_neighbors = neighbors[indices, 0]
    right_neighbors = neighbors[indices, 1]

    left_heights = heights[left_neighbors]
    right_heights = heights[right_neighbors]

    if adsorption and diffusion_direction != -1:
        raise ValueError(
            "adsorption and diffusion_direction cannot be set at the same time"
        )

    if adsorption:
        current_heights = heights[indices] + np.int32(1)
    else:
        current_heights = heights[indices]

    if diffusion_direction == 0:
        current_heights = heights[indices] + np.int32(1)
        right_heights = right_heights - np.int32(1)
    elif diffusion_direction == 1:
        current_heights = heights[indices] + np.int32(1)
        left_heights = left_heights - np.int32(1)

    num_occupied_direct = np.zeros_like(indices)
    num_occupied_diagonal = np.zeros_like(indices)

    # --- Direct Neighbors ---

    # 1. Neighbor Below: (Exist if height > 1)
    num_occupied_direct += current_heights > 1
    bottom_occupied = current_heights > 1

    # 2. Left Neighbor: (Exist if left_height >= current_height and > 0)
    num_occupied_direct += (current_heights <= left_heights) & (current_heights > 0)

    # 3. Right Neighbor: (Exist if right_height >= current_height and > 0)
    num_occupied_direct += (current_heights <= right_heights) & (current_heights > 0)

    # --- Diagonal Neighbors ---

    # 4. Upper-Left: (Exist if left_height >= current_height + 1, i.e., > current)
    num_occupied_diagonal += (current_heights < left_heights) & (current_heights >= 1)

    # 5. Upper-Right: (Exist if right_height > current)
    num_occupied_diagonal += (current_heights < right_heights) & (current_heights >= 1)

    # 6. Lower-Left:
    # Must be at height > 1 to have a lower neighbor (height 1 cannot have a lower neighbor particle)
    # Neighbor must be tall enough to fill the diagonal slot (h_left >= h - 1)
    has_lower_left = (current_heights > 1) & (left_heights >= (current_heights - 1))
    num_occupied_diagonal += has_lower_left

    # 7. Lower-Right:
    has_lower_right = (current_heights > 1) & (right_heights >= (current_heights - 1))
    num_occupied_diagonal += has_lower_right

    return num_occupied_direct, num_occupied_diagonal, bottom_occupied


class Lattice(BaseLattice):
    """1D+1 Lattice with height-based representation and periodic boundary conditions."""

    def __init__(self, model, lattice_size: int):
        """Initialize 1D+1 Lattice.

        Args:
            model: KMC Model instance.
            lattice_size (int): Number of sites in the lattice.
        """
        super().__init__(model=model)

        self.size = lattice_size
        self.heights = np.zeros(self.size, dtype=np.int32)
        self.neighbors_array = generate_neighbors_1d(self.size)

    def reset_lattice(self, lattice_size: int = None):
        """Reset lattice to new size.

        Args:
            lattice_size (int, optional): New size for the lattice. Defaults to None.
        """
        if lattice_size is not None:
            self.size = lattice_size
            self.neighbors_array = generate_neighbors_1d(self.size)

        self.heights = np.zeros(self.size, dtype=np.int32)

    def get_state(self) -> np.array:
        """Get the current lattice heights.

        Returns:
            np.array: Array of heights at each site.
        """
        return self.heights.copy()

    def set_state(self, heights: np.array):
        """Set lattice heights.

        Args:
            heights (np.array): Array of lattice heights.

        Raises:
            ValueError: If heights size doesn't match lattice size.
        """
        self.validate_size(self.size, heights.shape[0])
        self.heights = heights.astype(np.int32)
        self.model.rate_calculator.reset()

    def set_lattice_heights(self, heights: np.array):
        """Alias for set_state() for backward compatibility.

        Args:
            heights (np.array): Array of lattice heights.
        """
        self.set_state(heights)
