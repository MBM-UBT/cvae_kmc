import functools
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import cvae.data_handling.data_transforms as data_transforms


def _sort_key_for_pt_file(file_path: str) -> int:
    """
    Extract the numerical index from a filename to ensure deterministic sorting.

    Args:
        file_path: The full path or filename of the `.pt` file.

    Returns:
        The extracted integer index, or 0 if no digits are found.
    """
    name = os.path.basename(file_path)
    numbers = re.findall(r"\d+", name)
    return int(numbers[0]) if numbers else 0


def get_all_pt_files(data_dir: str) -> List[str]:
    """
    Recursively collect and deterministically sort all PyTorch data files in a directory.

    Args:
        data_dir: The root directory to search for trajectory files.

    Returns:
        A sorted list of absolute file paths ending in `.pt`.
    """
    pt_files: List[str] = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".pt"):
                pt_files.append(os.path.join(root, file_name))

    return sorted(pt_files, key=_sort_key_for_pt_file)


class KMC_Single_Dataset_Relative(Dataset):
    """
    Dataset for single-step Kinetic Monte Carlo (KMC) supervision.

    Loads cached trajectory files from disk and dynamically constructs
    input-target pairs representing a single time step (dt_step) evolution.
    """

    def __init__(
        self,
        data_dir: str,
        max_negative_change: int,
        max_positive_change: int,
        temp: bool = False,
        temp_range: Optional[Tuple[float, float]] = None,
        dt_step: int = 1,
        shift: bool = False,
        frac_sims: float = 1.0,
        max_steps: Optional[int] = None,
        test_mode: bool = False,
    ) -> None:
        """
        Initialize the dataset by validating and indexing available trajectory files.

        Args:
            data_dir: Root directory containing `.pt` simulation files.
            max_negative_change: Maximum allowed downward height change for normalization.
            max_positive_change: Maximum allowed upward height change for normalization.
            temp: If True, temperature conditioning data is extracted and passed to transforms.
            temp_range: Min and max temperature bounds for scaling.
            dt_step: The temporal distance (in steps) between the input and target states.
            shift: If True, applies random horizontal circular shifts for translational invariance.
            frac_sims: Fraction (0.0 to 1.0) of total available simulation files to load.
            max_steps: Artificial cap on the trajectory depth to use per file.
            test_mode: If True, just returns the lattices without applying transforms (for debugging/analysis).
        """
        self.dt_step = dt_step
        self.shift = shift
        self.temp = temp
        self.temp_range = temp_range
        self.max_negative_change = max_negative_change
        self.max_positive_change = max_positive_change
        self.test_mode = test_mode
        all_files = get_all_pt_files(data_dir)
        num_to_use = int(len(all_files) * frac_sims)
        self.files: List[str] = all_files[:num_to_use]

        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        # Peek into the first file to establish global tensor dimensions
        sample_data = torch.load(self.files[0], weights_only=False)
        self.depth: int = sample_data["target_trajectory"].shape[0]
        self.leaves: int = sample_data["target_trajectory"].shape[1]

        if max_steps is not None:
            self.depth = min(self.depth, max_steps)

        # The effective indexable depth is reduced because a corresponding target
        # state must exist dt_step ahead of the input state.
        self.effective_depth: int = self.depth - self.dt_step + 1

        if self.effective_depth <= 0:
            raise ValueError(
                f"dt_step ({self.dt_step}) is too large for the available "
                f"trajectory depth ({self.depth})."
            )

    def __len__(self) -> int:
        """
        Calculate the total number of valid input-target pairs across all files.

        Returns:
            Total indexable dataset size.
        """
        return len(self.files) * self.effective_depth * self.leaves

    @functools.lru_cache(maxsize=4)
    def _load_simulation_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load and cache a simulation file from disk.

        Using an LRU cache prevents severe I/O thrashing when the DataLoader
        randomly queries multiple indices that reside within the same physical file.
        """
        return torch.load(file_path, weights_only=False)

    def _decode_linear_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Decompose the flat 1D DataLoader index into specific 3D coordinates.

        Args:
            idx: The global sample index requested by the DataLoader.

        Returns:
            A tuple containing (file_index, depth_index, leaf_index).
        """
        elements_per_file = self.effective_depth * self.leaves

        file_idx = idx // elements_per_file
        remainder = idx % elements_per_file

        depth_idx = remainder // self.leaves
        leaf_idx = remainder % self.leaves

        return file_idx, depth_idx, leaf_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve and process a single simulation transition step.

        Args:
            idx: Flat dataset index.

        Returns:
            A tuple of (input_condition, delta_normalized, target_heights).
        """
        file_idx, depth_idx, leaf_idx = self._decode_linear_index(idx)

        data = self._load_simulation_file(self.files[file_idx])

        temperature_kelvin = None
        if self.temp:
            temperature_kelvin = data["metadata"]["temperature"]

        # Isolate the starting state and the future target state
        start_lattice = data["input_trajectory"][depth_idx].unsqueeze(0)
        target_step_idx = depth_idx + self.dt_step - 1
        final_lattice = data["target_trajectory"][target_step_idx, leaf_idx].unsqueeze(
            0
        )

        # If in test mode, skip all transformations and return raw lattices for analysis
        if self.test_mode:
            return (start_lattice, final_lattice)

        # Apply circular shift augmentation to enforce translational invariance
        if self.shift:
            shift_val = torch.randint(0, start_lattice.shape[1], size=(1,)).item()
            start_lattice = torch.roll(start_lattice, shifts=shift_val, dims=1)
            final_lattice = torch.roll(final_lattice, shifts=shift_val, dims=1)

        input_condition = data_transforms.create_input_condition(
            start_lattice,
            max_negative_change=self.max_negative_change,
            max_positive_change=self.max_positive_change,
            temp=self.temp,
            temperature_kelvin=temperature_kelvin,
            temp_range=self.temp_range,
        )

        delta_normalized, target = data_transforms.create_targets(
            start_lattice,
            final_lattice,
            max_negative_change=self.max_negative_change,
            max_positive_change=self.max_positive_change,
        )

        return (input_condition.squeeze(0), delta_normalized, target)
