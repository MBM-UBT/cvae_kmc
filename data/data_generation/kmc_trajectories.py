import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metrics import roughness_1d
from python_kmc import dendrite_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define tracked physical metrics to avoid boilerplate repetition
TRACKED_METRICS = ["growth", "ra", "rq", "rz", "dendrites", "kmc_steps"]

# Global model instance required for ProcessPoolExecutor to avoid
# severe overhead and pickling errors when passing the model between processes.
_worker_kmc_model: Any = None


class ProgressLogger:
    """Tracks and logs batch simulation progress with ETA estimations."""

    def __init__(
        self, total: int, desc: str = "Processing", log_file_path: Optional[str] = None
    ) -> None:
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.completed = 0
        self.last_reported_percent = 0
        self.log_file_path = log_file_path

        if self.log_file_path:
            with open(self.log_file_path, "a") as f:
                f.write(
                    f"\n--- Starting {self.desc} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )

    def _format_time(self, seconds: float) -> str:
        """Format raw seconds into a human-readable HH:MM:SS string."""
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def update(self, advance: int = 1) -> None:
        """Increment progress and emit logs if a 5% threshold is crossed."""
        self.completed += advance
        current_percent = int((self.completed / self.total) * 100)

        if (
            current_percent >= self.last_reported_percent + 5
            or self.completed == self.total
        ):
            self.last_reported_percent = (current_percent // 5) * 5

            elapsed = time.time() - self.start_time
            time_per_item = elapsed / self.completed if self.completed > 0 else 0
            est_remaining = (self.total - self.completed) * time_per_item

            log_msg = (
                f"[{self.desc}] {current_percent:3d}% ({self.completed}/{self.total}) "
                f"| Elapsed: {self._format_time(elapsed)} "
                f"| Est. Remaining: {self._format_time(est_remaining)}"
            )

            print(log_msg)
            if self.log_file_path:
                with open(self.log_file_path, "a") as f:
                    f.write(log_msg + "\n")


def init_worker(lattice_size: int) -> None:
    """Initialize the global KMC model for an isolated worker process."""
    global _worker_kmc_model
    _worker_kmc_model = dendrite_model.generate_kmc_model(lattice_size=lattice_size)


def run_single_kmc(
    sim_idx: int,
    start_lattice: np.ndarray,
    steps_count: int,
    dt_val: float,
    temp_val: float,
    run_seed: int,
    temp_change_per_step: float,
    temp_jumps_every_n_steps: int,
    temp_jump_height: float,
) -> Tuple[int, Dict[str, List[float]]]:
    """Execute a single deterministic KMC trajectory and track physical metrics."""
    global _worker_kmc_model

    # Enforce deterministic behavior and reset environmental conditions
    _worker_kmc_model.set_seed(run_seed)
    _worker_kmc_model.rate_calculator.temperature_kelvin = temp_val
    _worker_kmc_model.lattice.set_lattice_heights(start_lattice.copy())

    metrics: Dict[str, List[float]] = {m: [] for m in TRACKED_METRICS}

    for step in range(steps_count):
        start_steps = _worker_kmc_model.loop_counter
        start_mass = np.sum(_worker_kmc_model.lattice.heights)

        _worker_kmc_model.run_simulation_dt(dt_val)

        # Apply continuous or discrete temperature dynamics
        current_temp = _worker_kmc_model.rate_calculator.temperature_kelvin
        if temp_change_per_step != 0.0:
            _worker_kmc_model.update_temperature(
                temperature_kelvin=current_temp + temp_change_per_step
            )

        if temp_jumps_every_n_steps > 0 and (step + 1) % temp_jumps_every_n_steps == 0:
            _worker_kmc_model.update_temperature(
                temperature_kelvin=current_temp + temp_jump_height
            )

        current_heights = _worker_kmc_model.lattice.heights

        metrics["growth"].append(np.sum(current_heights) - start_mass)
        metrics["ra"].append(roughness_1d.Ra(current_heights))
        metrics["rq"].append(roughness_1d.Rq(current_heights))
        metrics["rz"].append(roughness_1d.Rz(current_heights))
        metrics["dendrites"].append(roughness_1d.count_dendrites(current_heights))
        metrics["kmc_steps"].append(_worker_kmc_model.loop_counter - start_steps)

    return sim_idx, metrics


def _build_start_lattice(args: argparse.Namespace) -> np.ndarray:
    """Construct the initial surface boundary conditions based on test requirements."""
    dummy_model = dendrite_model.generate_kmc_model(lattice_size=args.lattice_size)
    lattice = np.zeros_like(dummy_model.lattice.heights, dtype=int)

    if args.random_start:
        lattice = np.load(os.path.join(BASE_DIR, "random_start.npy"))

    if args.ood_test:
        center = args.lattice_size // 2
        lattice[center - 5 : center + 5] = args.dendrite_height

    return lattice


def _generate_filenames(
    args: argparse.Namespace, temp: Optional[float] = None
) -> Tuple[str, str]:
    """Generate deterministic log and output filenames based on simulation parameters."""
    base_name = f"L{args.lattice_size}_{args.total_simulations}sims"

    if args.random_start:
        condition_str = "random_start"
    elif args.ood_test:
        condition_str = f"ood_test_{args.dendrite_height}"
    else:
        condition_str = "empty_start"

    # Log filename represents the whole batch
    log_file = f"run_kmc_{base_name}_{len(args.temperature)}temps_{condition_str}.log"

    # NPZ filename is specific to a temperature split
    if temp is not None:
        out_file = f"{args.val_or_test}_kmc_{args.lattice_size}_{args.total_simulations}_samples_{condition_str}_T{int(temp)}_{args.steps}_steps_seed{args.base_seed}_dt{args.dt}.npz"
    else:
        out_file = ""

    return log_file, out_file


def _setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trajectory generation for KMC model."
    )
    parser.add_argument("--total_simulations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[300.0],
        help="List of temperatures in Kelvin",
    )
    parser.add_argument("--dt", type=float, default=5e-6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--val_or_test", type=str, default="val", choices=["val", "test"]
    )
    parser.add_argument(
        "--ood_test",
        type=int,
        default=0,
        help="Generate an artificial dendrite for Out-Of-Distribution testing.",
    )
    parser.add_argument("--dendrite_height", type=int, default=30)
    parser.add_argument("--random_start", type=int, default=0)
    parser.add_argument("--temp_change_per_dt", type=float, default=0.0)
    parser.add_argument("--temp_jumps_every_n_steps", type=int, default=0)
    parser.add_argument("--temp_jump_height", type=int, default=0)
    return parser


def main(parsed_args: Optional[argparse.Namespace] = None) -> None:
    if parsed_args is None:
        args = _setup_argument_parser().parse_args()
    else:
        args = parsed_args

    log_dir = os.path.join(BASE_DIR, "logging")
    results_dir = os.path.join(BASE_DIR, "..", "kmc_trajectories")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_filename, _ = _generate_filenames(args)
    full_log_path = os.path.join(log_dir, log_filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    with open(full_log_path, "w") as f:
        f.write(f"KMC Simulation Run: {timestamp}\n")
        f.write(
            f"Total Simulations per Temp: {args.total_simulations}\nSteps: {args.steps}\n"
        )
        f.write(f"Temperatures to process: {args.temperature}K\n")
        f.write(f"Base Seed: {args.base_seed}\n")
        f.write("=" * 40 + "\n")

    start_lattice = _build_start_lattice(args)

    for temp in args.temperature:
        print(f"\n>>> Starting KMC batch for Temperature: {temp}K <<<")

        # Initialize aggregate tracking dictionaries
        batch_metrics: Dict[str, List[Any]] = {
            m: [None] * args.total_simulations for m in TRACKED_METRICS
        }

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(args.lattice_size,),
        ) as executor:

            # Using base_seed + i ensures that simulation N at Temp 1 has the exact
            # same initial random state as simulation N at Temp 2, enabling paired testing.
            futures = [
                executor.submit(
                    run_single_kmc,
                    i,
                    start_lattice,
                    args.steps,
                    args.dt,
                    temp,
                    args.base_seed + i,
                    args.temp_change_per_dt,
                    args.temp_jumps_every_n_steps,
                    args.temp_jump_height,
                )
                for i in range(args.total_simulations)
            ]

            logger = ProgressLogger(
                total=args.total_simulations,
                desc=f"KMC T={int(temp)}K",
                log_file_path=full_log_path,
            )

            # Collect results asynchronously as they complete
            for future in as_completed(futures):
                sim_idx, metrics = future.result()
                for m in TRACKED_METRICS:
                    batch_metrics[m][sim_idx] = metrics[m]
                logger.update()

        # Compile and save batch data
        metadata = {
            "temperature_K": temp,
            "dt": args.dt,
            "steps": args.steps,
            "lattice_size": args.lattice_size,
            "base_seed": args.base_seed,
            "total_simulations": args.total_simulations,
            "timestamp": timestamp,
        }

        _, out_filename = _generate_filenames(args, temp)
        out_filepath = os.path.join(results_dir, out_filename)

        np.savez(
            out_filepath,
            **{m: np.array(batch_metrics[m]) for m in TRACKED_METRICS},
            start_lattice=start_lattice,
            metadata=np.array(metadata, dtype=object),
        )
        print(f"Saved KMC results and metadata for {temp}K to: {out_filename}")


if __name__ == "__main__":
    main()
