"""
Generates data for 1D+1 KMC simulations in parallel.
Each simulation trajectory is saved as a single consolidated PyTorch data file.
"""

import argparse
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import python_kmc.dendrite_model as dendrite_model

# Constants
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SEED_SCALING_FACTOR = 100000
PROGRESS_LOGGING_CHUNKS = 20
DEFAULT_MAX_WORKERS = 127


def generate_1d_start_lattice(
    lattice_width: int,
    empty_probability: float = 0.3,
    absolute_max_height: int = 50,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a 1D+1 starting lattice with randomized localized heights.

    Args:
        lattice_width: Number of discrete sites in the 1D lattice.
        empty_probability: Probability (0.0 to 1.0) that the entire lattice remains flat.
        absolute_max_height: Upper bound for random height initialization.
        seed: Optional RNG seed for deterministic initialization.

    Returns:
        A 1D tensor representing initial surface heights.
    """
    if seed is not None:
        np.random.seed(seed)

    lattice = np.zeros(lattice_width, dtype=np.float32)

    if np.random.rand() < empty_probability:
        return lattice

    coverage_percentage = np.random.rand()
    mask = np.random.rand(lattice_width) < coverage_percentage
    num_spots_to_fill = mask.sum()

    if num_spots_to_fill > 0:
        random_heights = np.random.randint(
            low=1,
            high=np.random.randint(2, absolute_max_height + 1),
            size=(num_spots_to_fill,),
        )
        lattice[mask] = random_heights

    return lattice


def _parse_float_or_none(value: str) -> Optional[float]:
    """
    Parse command line arguments that can accept either a numerical float or an explicit 'None' string.
    """
    if str(value).lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float or 'None'")


def _get_save_directory(split_dir: str, temp: float, use_subdir: bool) -> str:
    """
    Determine the target output directory, conditionally grouping by temperature.
    """
    if use_subdir:
        save_dir = os.path.join(split_dir, f"T_{temp:.0f}")
    else:
        save_dir = split_dir

    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _initialize_and_warmup_model(
    lattice_size: int,
    temp: float,
    sample_idx: int,
    warmup_particles: int,
    random_start: int,
    empty_probability: float,
    max_height: int,
    dt: float,
) -> Tuple[Any, np.ndarray]:
    """
    Instantiate the KMC model, set deterministic seeds, and perform initial warm-up deposits.

    A short simulation is explicitly run if starting from a randomized lattice to allow
    the system to naturally relax and escape artificially high-energy initial states
    before the recorded trajectory begins.
    """
    sim_model = dendrite_model.generate_kmc_model(lattice_size=lattice_size)
    sim_model.rate_calculator.temperature_kelvin = temp

    # Scale the float temperature to an integer to guarantee unique seeds across varying temperature ranges
    deterministic_seed = sample_idx + int(temp * SEED_SCALING_FACTOR)
    sim_model.set_seed(deterministic_seed)
    sim_model.reset()

    while np.sum(sim_model.lattice.heights) < warmup_particles:
        sim_model.run_simulation_steps(steps=1)

    start_lattice = sim_model.lattice.heights.copy()

    if random_start > 0:
        start_lattice = generate_1d_start_lattice(
            lattice_width=lattice_size,
            empty_probability=empty_probability,
            absolute_max_height=max_height,
        )

        if np.sum(start_lattice) != 0:
            sim_model.run_simulation_dt(dt=dt)
            start_lattice = sim_model.lattice.heights.copy()

    return sim_model, start_lattice


def _generate_trajectory(
    sim_model: Any,
    start_lattice: np.ndarray,
    depth: int,
    leaves: int,
    dt: float,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[List[int]]]:
    """
    Execute the core KMC rollout, capturing sequential lattice states and loop metrics.
    """
    history_input_lattices = []
    history_target_lattices = []
    history_loops = []

    for _ in range(depth):
        current_step_input = start_lattice.copy()
        history_input_lattices.append(current_step_input)

        step_targets = []
        step_loops = []

        for _ in range(leaves):
            sim_model.reset(heights=current_step_input)
            sim_model.run_simulation_dt(dt=dt)

            step_targets.append(sim_model.lattice.heights.copy())
            step_loops.append(sim_model.loop_counter)

        history_target_lattices.append(step_targets)
        history_loops.append(step_loops)
        start_lattice = step_targets[0].copy()

    return history_input_lattices, history_target_lattices, history_loops


def _save_trajectory(
    save_path: str,
    history_inputs: List[np.ndarray],
    history_targets: List[List[np.ndarray]],
    history_loops: List[List[int]],
    metadata: Dict[str, Union[float, int]],
) -> None:
    """
    Serialize the generated trajectory tensors and metadata to disk.
    """
    try:
        torch.save(
            {
                "input_trajectory": torch.from_numpy(np.array(history_inputs)).float(),
                "target_trajectory": torch.from_numpy(
                    np.array(history_targets)
                ).float(),
                "loops": torch.from_numpy(np.array(history_loops)).float(),
                "metadata": metadata,
            },
            save_path,
        )
    except Exception as e:
        print(f"Error saving {save_path}: {e}")


def process_sample(params: Tuple[int, float, str, bool, argparse.Namespace]) -> int:
    """
    Worker function to orchestrate the end-to-end generation of a single simulation sample.
    """
    sample_idx, temp, split_dir, use_subdir, args = params

    save_dir = _get_save_directory(split_dir, temp, use_subdir)
    save_path = os.path.join(save_dir, f"sim_{sample_idx}.pt")

    sim_model, start_lattice = _initialize_and_warmup_model(
        lattice_size=args.lattice_size,
        temp=temp,
        sample_idx=sample_idx,
        warmup_particles=args.warmup,
        random_start=args.random_start,
        empty_probability=args.empty_probability,
        max_height=args.max_height,
        dt=args.dt,
    )

    history_inputs, history_targets, history_loops = _generate_trajectory(
        sim_model=sim_model,
        start_lattice=start_lattice,
        depth=args.depth,
        leaves=args.leaves,
        dt=args.dt,
    )

    metadata = {
        "temperature": temp,
        "lattice_size": args.lattice_size,
        "dt": args.dt,
        "depth": args.depth,
        "leaves": args.leaves,
    }

    _save_trajectory(
        save_path, history_inputs, history_targets, history_loops, metadata
    )

    return sample_idx


def _setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure and return the CLI argument parser.
    """
    parser = argparse.ArgumentParser(description="KMC Data Generator")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=os.path.join(BASE_DIR, "..", "training_datasets", "test"),
    )
    parser.add_argument("--lattice_size", type=int, default=256)
    parser.add_argument("--dt", type=float, default=5e-6)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--leaves", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--possible_temps", nargs="+", type=float, default=[300])
    parser.add_argument(
        "--temp_range", nargs="+", type=_parse_float_or_none, default=None
    )
    parser.add_argument("--samples_per_temp", type=int, default=100)
    parser.add_argument("--train_sims", type=int, default=80)
    parser.add_argument("--validation_sims", type=int, default=10)
    parser.add_argument("--test_sims", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="test.log")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--random_start", type=int, default=0)
    parser.add_argument("--empty_probability", type=float, default=0.0)
    parser.add_argument("--max_height", type=int, default=0)
    return parser


def main(parsed_args: Optional[argparse.Namespace] = None) -> None:
    """
    Main execution pipeline for trajectory generation across multiprocessing pools.
    """
    if parsed_args is None:
        parser = _setup_argument_parser()
        args = parser.parse_args()
    else:
        args = parsed_args

    train_dir = os.path.join(args.output_dir, "train")
    validation_dir = os.path.join(args.output_dir, "validation")
    test_dir = os.path.join(args.output_dir, "test")

    for directory in [train_dir, validation_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    total_count = args.samples_per_temp
    train_end = args.train_sims
    val_end = train_end + args.validation_sims

    tasks = []

    is_continuous_mode = args.temp_range is not None and None not in args.temp_range

    if is_continuous_mode:
        use_subdirs = False
        t_min, t_max = args.temp_range
        logging_msg = f"Generating {total_count} trajectories sampling uniformly in range [{t_min}, {t_max}]. No subfolders."

        for idx in range(total_count):
            temp = float(np.random.uniform(t_min, t_max))
            split = (
                train_dir
                if idx < train_end
                else (validation_dir if idx < val_end else test_dir)
            )
            tasks.append((idx, temp, split, use_subdirs, args))

    else:
        use_subdirs = True
        logging_msg = f"Generating trajectories for discrete temperatures: {args.possible_temps}. Using subfolders."

        for temp in args.possible_temps:
            for idx in range(total_count):
                split = (
                    train_dir
                    if idx < train_end
                    else (validation_dir if idx < val_end else test_dir)
                )
                tasks.append((idx, temp, split, use_subdirs, args))

    system_cpus = multiprocessing.cpu_count() - 1
    num_workers = args.num_workers or min(system_cpus, DEFAULT_MAX_WORKERS)

    log_file_dir = os.path.join(BASE_DIR, "logging")
    os.makedirs(log_file_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_file_dir, args.log_file),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w",
    )
    logging.info(logging_msg)

    with multiprocessing.Pool(processes=num_workers) as pool:
        logging_interval = max(1, len(tasks) // PROGRESS_LOGGING_CHUNKS)
        for i, _ in enumerate(pool.imap_unordered(process_sample, tasks), 1):
            if i % logging_interval == 0:
                logging.info(f"Progress: {i/len(tasks)*100:.0f}% ({i}/{len(tasks)})")

    logging.info("Generation Complete.")


if __name__ == "__main__":
    main()
