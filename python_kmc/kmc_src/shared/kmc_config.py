"""Shared KMC configuration and constants.

This module centralizes configuration constants used across all KMC models.
"""

import numpy as np

# Default number of processes for 1D+1 model
NUM_PROCESSES_1D1 = 4  # adsorption, desorption, diffusion left, diffusion right

# Numba JIT decorator control
USE_NUMBA_1D1 = True  # Set to False if you encounter numba compatibility issues

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-10

# Default data types
LATTICE_DTYPE = np.int32
RATE_DTYPE = np.float32
ENERGY_DTYPE = np.float32
