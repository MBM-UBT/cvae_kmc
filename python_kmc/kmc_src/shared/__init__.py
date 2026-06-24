"""Shared KMC modules for dimension-agnostic code.

This package provides base classes, utilities, and factories for all KMC implementations.
"""

from .kmc_config import FLOAT_TOLERANCE, NUM_PROCESSES_1D1, USE_NUMBA_1D1
from .kmc_model_base import BaseKMCModel
from .lattice_base import BaseLattice
from .neighbor_generation import generate_neighbors_1d, generate_neighbors_2d
from .process_handler_base import ProcessHandlerBase, select_process_numba
from .rate_calculator_base import BaseRateCalculator

__all__ = [
    # Base classes
    "BaseKMCModel",
    "BaseLattice",
    "ProcessHandlerBase",
    "BaseRateCalculator",
    # Functions
    "select_process_numba",
    "generate_neighbors_1d",
    "generate_neighbors_2d",
    "create_kmc_model",
    # Constants
    "NUM_PROCESSES_1D1",
    "USE_NUMBA_1D1",
    "FLOAT_TOLERANCE",
]
