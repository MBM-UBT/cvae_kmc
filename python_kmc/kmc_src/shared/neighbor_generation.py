"""Shared neighbor generation utilities for KMC lattices.

This module provides dimension-specific neighbor array generation functions
that can be used by any KMC model.
"""

import numpy as np
from numba import njit


@njit
def generate_neighbors_1d(sites: int) -> np.array:
    """Generate neighbor connections for 1D lattice with periodic boundary conditions.

    Args:
        sites (int): Number of sites in the 1D lattice.

    Returns:
        np.array: An (sites, 2) array where neighbors[i] = [left_neighbor, right_neighbor]
    """
    neighbors = np.zeros((sites, 2), dtype=np.int32)

    for idx in range(sites):
        neighbors[idx, 0] = (idx - 1) % sites  # Left neighbor
        neighbors[idx, 1] = (idx + 1) % sites  # Right neighbor

    return neighbors


def generate_neighbors_2d(sites_x: int, sites_y: int) -> np.array:
    """Generate neighbor connections for 2D lattice with periodic boundary conditions.

    The 2D grid is flattened to a 1D array of size (sites_x * sites_y).
    Neighbors are indexed as: 0=up, 1=right, 2=down, 3=left, 4=upper-right,
    5=lower-right, 6=lower-left, 7=upper-left.

    Args:
        sites_x (int): Number of sites in x direction.
        sites_y (int): Number of sites in y direction.

    Returns:
        np.array: An (N, 8) array containing neighbor indices for each site.
    """
    total_sites = sites_x * sites_y
    neighbors = np.zeros((total_sites, 8), dtype=np.int32)

    for y in range(sites_y):
        for x in range(sites_x):
            idx = y * sites_x + x

            # Periodic boundary conditions
            y_up = (y + 1) % sites_y
            y_down = (y - 1) % sites_y
            x_right = (x + 1) % sites_x
            x_left = (x - 1) % sites_x

            # Direct neighbors
            neighbors[idx, 0] = y_up * sites_x + x  # Up
            neighbors[idx, 1] = y * sites_x + x_right  # Right
            neighbors[idx, 2] = y_down * sites_x + x  # Down
            neighbors[idx, 3] = y * sites_x + x_left  # Left

            # Diagonal neighbors
            neighbors[idx, 4] = y_up * sites_x + x_right  # Upper-Right
            neighbors[idx, 5] = y_down * sites_x + x_right  # Lower-Right
            neighbors[idx, 6] = y_down * sites_x + x_left  # Lower-Left
            neighbors[idx, 7] = y_up * sites_x + x_left  # Upper-Left

    return neighbors
