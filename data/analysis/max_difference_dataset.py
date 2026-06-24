import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cvae.data_handling.data_loaders as data_loaders

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    data_dir = os.path.join(
        BASE_DIR,
        "..",
        "training_datasets",
        "single_256_10000_sims_200_steps_dt5e-06_temp_range",  # enter the name of the dataset you want to analyze here
        "train",
    )

    dataset = data_loaders.KMC_Single_Dataset_Relative(
        data_dir=data_dir,
        dt_step=1,
        test_mode=True,
        max_negative_change=10,  # just set some values here, these are ignored in test mode
        max_positive_change=10,
    )
    data_loader = DataLoader(dataset, batch_size=512, num_workers=10, shuffle=False)

    # Initialize with infinity to ensure correct min/max tracking even if all values
    # happen to be strictly negative or strictly positive.
    global_max_diff = float("-inf")
    global_min_diff = float("inf")
    global_max_diff_one_profile = float("-inf")
    global_min_diff_one_profile = float("inf")

    for data in tqdm(data_loader):
        input_trajectory = data[0]
        target_trajectory = data[1]

        # 1. Vectorized calculations over the entire batch at once
        # torch.diff works natively on batched tensors (computes along the last dimension by default)
        diff_one_profile = torch.diff(target_trajectory)
        diff = target_trajectory - input_trajectory

        # 2. Find the max/min of the entire batch tensor
        batch_max_diff_one = torch.max(diff_one_profile).item()
        batch_min_diff_one = torch.min(diff_one_profile).item()
        batch_max_diff = torch.max(diff).item()
        batch_min_diff = torch.min(diff).item()

        # 3. Update global values using standard Python max/min
        global_max_diff_one_profile = max(
            global_max_diff_one_profile, batch_max_diff_one
        )
        global_min_diff_one_profile = min(
            global_min_diff_one_profile, batch_min_diff_one
        )
        global_max_diff = max(global_max_diff, batch_max_diff)
        global_min_diff = min(global_min_diff, batch_min_diff)

    # difference between adjacent sites
    print(f"Global max difference (adjacent sites): {global_max_diff_one_profile}")
    print(f"Global min difference (adjacent sites): {global_min_diff_one_profile}")

    # difference between same site across time steps
    print(f"Global max difference (same site, different times): {global_max_diff}")
    print(f"Global min difference (same site, different times): {global_min_diff}")
