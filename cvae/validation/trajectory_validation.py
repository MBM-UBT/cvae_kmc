import os
from typing import Dict, List, Union

import numpy as np
import torch
    
import metrics.roughness_1d as roughness_1d

# Define the physical metrics we track to avoid variable duplication
TRACKED_METRICS = ["growth", "rq"]


def _calculate_nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between target and prediction.

    Using relative error scaling allows us to compare errors across physical metrics
    with completely different absolute scales (e.g., lattice mass vs. nanometers).
    """
    return float(np.sqrt(np.mean(((target - prediction) / target) ** 2)))


def trajectory_val_loss(
    kmc_data_path: str,
    ml_model: torch.nn.Module,
    pass_temp_to_ml_model: bool = False,
    return_errors_only: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compare a trained model's autoregressive rollout against ground truth KMC trajectories
    by calculating the deviation in mean and standard deviation of key surface metrics.
    """
    device = next(ml_model.parameters()).device

    if os.path.isdir(kmc_data_path):
        file_paths = [
            os.path.join(kmc_data_path, f)
            for f in os.listdir(kmc_data_path)
            if os.path.isfile(os.path.join(kmc_data_path, f))
        ]
    else:
        file_paths = [kmc_data_path]

    # Dynamically generate tracking lists using list comprehensions
    errors_list: Dict[str, List[float]] = {
        f"nrmse_{stat}_{m}": [] for stat in ["mean", "std"] for m in TRACKED_METRICS
    }
    arrays_list: Dict[str, List[np.ndarray]] = {
        f"{stat}_{m}_{source}": []
        for stat in ["mean", "std"]
        for m in TRACKED_METRICS
        for source in ["kmc", "ml"]
    }

    for f_path in file_paths:
        kmc_data = np.load(f_path, allow_pickle=True)
        metadata = kmc_data["metadata"].item()
        temperature_kelvin = metadata["temperature_K"]
        lattice_size = metadata["lattice_size"]

        # 1. Load KMC baseline metrics
        kmc_metrics = {m: kmc_data[m] for m in TRACKED_METRICS}
        num_sims, num_steps = kmc_metrics["growth"].shape

        # Calculate baseline KMC statistics across the batch/simulation dimension
        for m in TRACKED_METRICS:
            arrays_list[f"mean_{m}_kmc"].append(np.mean(kmc_metrics[m], axis=0))
            arrays_list[f"std_{m}_kmc"].append(np.std(kmc_metrics[m], axis=0))

        # 2. Setup ML Simulation
        ml_metrics = {m: np.zeros((num_sims, num_steps)) for m in TRACKED_METRICS}

        # Start from a flat lattice. Expanded to match the batch size of the KMC data
        # so we evaluate the exact same number of stochastic rollouts.
        start_lattice = np.zeros((lattice_size,), dtype=int)
        ml_input = (
            torch.from_numpy(start_lattice).float().unsqueeze(0).unsqueeze(0).to(device)
        ).expand(num_sims, 1, -1)

        # 3. Autoregressive ML Rollout Loop
        for step in range(num_steps):
            start_mass = torch.sum(ml_input, dim=-1).squeeze(1)

            with torch.no_grad():
                ml_output = ml_model.sample(
                    n_samples=1,
                    device=device,
                    input_heights_int=ml_input,
                    argmax=False,
                    tau=1.0,
                    top_p=1.0,
                    p_min=0.0,
                    temperature_kelvin=(
                        temperature_kelvin if pass_temp_to_ml_model else None
                    ),
                )[0]

            # Extract and store per-step properties
            ml_metrics["growth"][:, step] = (
                (torch.sum(ml_output, dim=-1).squeeze(1) - start_mass).cpu().numpy()
            )
            ml_metrics["rq"][:, step] = (
                roughness_1d.Rq(ml_output).squeeze(1).cpu().numpy()
            )

            ml_input = ml_output

        # 4. Compute ML statistics and compare against KMC targets
        for m in TRACKED_METRICS:
            mean_ml = np.mean(ml_metrics[m], axis=0)
            std_ml = np.std(ml_metrics[m], axis=0)

            arrays_list[f"mean_{m}_ml"].append(mean_ml)
            arrays_list[f"std_{m}_ml"].append(std_ml)

            # Retrieve the most recently appended KMC target to calculate NRMSE
            target_mean_kmc = arrays_list[f"mean_{m}_kmc"][-1]
            target_std_kmc = arrays_list[f"std_{m}_kmc"][-1]

            errors_list[f"nrmse_mean_{m}"].append(
                _calculate_nrmse(target_mean_kmc, mean_ml)
            )
            errors_list[f"nrmse_std_{m}"].append(
                _calculate_nrmse(target_std_kmc, std_ml)
            )

    # Aggregate final output
    final_results: Dict[str, Union[float, np.ndarray]] = {
        k: float(np.mean(v)) for k, v in errors_list.items()
    }

    if not return_errors_only:
        for k, v in arrays_list.items():
            final_results[k] = np.stack(v, axis=0)

    return final_results
