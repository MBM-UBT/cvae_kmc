import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from cvae.model import architecture
from metrics import roughness_1d
from python_kmc import dendrite_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define tracked physical metrics to prevent variable duplication
TRACKED_METRICS = ["growth", "ra", "rq", "rz", "dendrites"]


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


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Configure and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run trajectory generation for ML model."
    )
    parser.add_argument("--total_simulations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[300.0],
        help="List of base temperatures in Kelvin",
    )
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--ml_model_name", type=str, default="isothermal_model")
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--pass_temp_to_ml", type=int, default=0)
    parser.add_argument(
        "--ood_test",
        type=int,
        default=0,
        help="Generate a massive dendrite for Out-Of-Distribution testing.",
    )
    parser.add_argument("--dendrite_height", type=int, default=30)
    parser.add_argument("--random_start", type=int, default=0)
    parser.add_argument("--temp_change_per_dt", type=float, default=0.0)
    parser.add_argument("--temp_jumps_every_n_steps", type=int, default=0)
    parser.add_argument("--temp_jump_height", type=int, default=10)
    return parser


def _build_start_lattice(args: argparse.Namespace) -> np.ndarray:
    """Construct the initial surface boundary conditions based on test requirements."""
    dummy_model = dendrite_model.generate_kmc_model(lattice_size=args.lattice_size)
    lattice = dummy_model.lattice.heights.copy()

    if args.ood_test:
        center = args.lattice_size // 2
        lattice[center - 5 : center + 5] = args.dendrite_height

    if args.random_start:
        lattice = np.load(os.path.join(BASE_DIR, "random_start.npy"))

    return lattice


def _generate_filenames(
    args: argparse.Namespace, base_temp: Optional[float] = None
) -> Tuple[str, str]:
    """Generate deterministic log and output filenames based on simulation parameters."""
    base_name = (
        f"{args.ml_model_name}_L{args.lattice_size}_{args.total_simulations}sims"
    )

    if args.random_start:
        condition_str = "random_start"
    elif args.ood_test:
        condition_str = f"ood_test_{args.dendrite_height}"
    else:
        condition_str = "empty_start"

    # Log filename represents the whole batch
    log_file = f"run_{base_name}_{len(args.temperature)}temps_{condition_str}.log"

    # NPZ filename is specific to a temperature split
    if base_temp is not None:
        out_file = f"{args.ml_model_name}_{args.lattice_size}_{args.total_simulations}_samples_{condition_str}_T{int(base_temp)}_{args.steps}_steps.npz"
    else:
        out_file = ""

    return log_file, out_file


def _load_ml_model(
    args: argparse.Namespace, device: torch.device
) -> architecture.VariationalAutoencoder_FullyConv:
    """Load the pre-trained CVAE model architecture and weights."""
    path_to_model = (
        os.path.join(args.model_folder, args.ml_model_name)
        if args.model_folder
        else args.ml_model_name
    )
    model_dir = os.path.join(BASE_DIR, "../../cvae/trained_models", path_to_model)

    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "best_model.pt")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    ml_model = architecture.VariationalAutoencoder_FullyConv(**model_config)

    # Strict=False allows flexibility if the model was saved with non-critical architectural variations
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    ml_model.load_state_dict(state_dict, strict=False)

    ml_model = ml_model.to(device)
    ml_model.eval()

    return ml_model


def main(parsed_args: Optional[argparse.Namespace] = None) -> None:
    """Main execution pipeline for generating neural network lattice trajectories."""
    if parsed_args is None:
        args = _setup_argument_parser().parse_args()
    else:
        args = parsed_args

    log_dir = os.path.join(BASE_DIR, "logging")
    results_dir = os.path.join(BASE_DIR, "..", "cvae_trajectories")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_filename, _ = _generate_filenames(args)
    full_log_path = os.path.join(log_dir, log_filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    with open(full_log_path, "w") as f:
        f.write(f"ML Generation Run: {timestamp}\n")
        f.write(
            f"Total Simulations per Temp: {args.total_simulations}\nSteps: {args.steps}\n"
        )
        f.write(f"Temperatures to process: {args.temperature}K\n")
        f.write("=" * 40 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_lattice = _build_start_lattice(args)
    ml_model = _load_ml_model(args, device)

    # Batch Process across all requested base temperatures
    for base_temp in args.temperature:
        print(f"\n>>> Starting ML batch for Temperature: {base_temp}K <<<")

        logger = ProgressLogger(
            total=args.steps,  # Tracking steps instead of simulations as the batch is processed simultaneously
            desc=f"ML T={int(base_temp)}K",
            log_file_path=full_log_path,
        )

        # Pre-allocate batched metric arrays
        ml_metrics: Dict[str, np.ndarray] = {
            m: np.zeros((args.total_simulations, args.steps)) for m in TRACKED_METRICS
        }

        # Initialize network inputs. We duplicate the starting lattice across the batch dimension.
        ml_input = (
            torch.from_numpy(start_lattice).float().unsqueeze(0).unsqueeze(0).to(device)
        ).expand(args.total_simulations, 1, -1)

        current_temp = base_temp

        # Autoregressive generation loop
        for step in range(args.steps):
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
                    temperature_kelvin=(current_temp if args.pass_temp_to_ml else None),
                )[0]

            # Process dynamic temperature environments
            if args.temp_change_per_dt != 0.0:
                current_temp += args.temp_change_per_dt

            if (
                args.temp_jumps_every_n_steps > 0
                and (step + 1) % args.temp_jumps_every_n_steps == 0
            ):
                current_temp += args.temp_jump_height

            # Extract metrics directly to CPU numpy arrays
            growth = torch.sum(ml_output, dim=-1).squeeze(1) - start_mass

            ml_metrics["growth"][:, step] = growth.cpu().numpy()
            ml_metrics["ra"][:, step] = (
                roughness_1d.Ra(ml_output).squeeze(1).cpu().numpy()
            )
            ml_metrics["rq"][:, step] = (
                roughness_1d.Rq(ml_output).squeeze(1).cpu().numpy()
            )
            ml_metrics["rz"][:, step] = (
                roughness_1d.Rz(ml_output).squeeze(1).cpu().numpy()
            )
            ml_metrics["dendrites"][:, step] = (
                roughness_1d.count_dendrites(ml_output).squeeze(1).cpu().numpy()
            )

            ml_input = ml_output
            logger.update()

        # Compile and save batch data
        metadata = {
            "temperature_K": base_temp,  # Ensure we save the original base temperature
            "steps": args.steps,
            "lattice_size": args.lattice_size,
            "total_simulations": args.total_simulations,
            "timestamp": timestamp,
        }

        _, out_filename = _generate_filenames(args, base_temp)
        out_filepath = os.path.join(results_dir, out_filename)

        np.savez(
            out_filepath,
            **ml_metrics,
            start_lattice=start_lattice,
            metadata=np.array(metadata, dtype=object),
        )
        print(f"Saved ML results and metadata for {base_temp}K to: {out_filename}")


if __name__ == "__main__":
    main()
