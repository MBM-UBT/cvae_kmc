import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import cvae.model.architecture as architecture
import cvae.validation.trajectory_validation as trajectory_validation
from cvae.data_handling.data_loaders import KMC_Single_Dataset_Relative

# Enable TF32 for hardware-accelerated matrix multiplications on Ampere+ GPUs
# This yields significant speedups for convolutions and matmuls with minimal precision loss.
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
except AttributeError:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Constants
BYTES_PER_MEGABYTE = 1024**2
VAL_LOSS_GROWTH_WEIGHT = 0.95
VAL_LOSS_STD_WEIGHT = 0.05
MODEL_GROUPS = 8


def setup_logger(
    name: str, log_dir: str, is_main_process: bool, console_only: bool = False
) -> logging.Logger:
    """
    Create the shared training logger for console and optional file output.

    Args:
        name: Name of the logging instance.
        log_dir: Directory to save the .log file.
        is_main_process: Flag to restrict file writing to the primary node in DDP.
        console_only: If True, bypasses file creation entirely.

    Returns:
        A configured standard Python logger.
    """
    is_saving_enabled = (console_only is False) and is_main_process

    if is_saving_enabled:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("CVAE_Training")
    logger.setLevel(logging.INFO if is_main_process else logging.ERROR)

    if logger.hasHandlers():
        logger.handlers.clear()

    if is_main_process:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        if is_saving_enabled:
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for training and evaluation runs.

    Args:
        args: Optional list of string arguments (defaults to sys.argv).

    Returns:
        Namespace object containing all parsed CLI configurations.
    """
    default_train_dir = os.path.join(
        BASE_DIR,
        "../..",
        "data",
        "training_datasets",
        "single_256_5000_sims_200_steps_dt5e-06",
        "train",
    )
    default_val_data_path = os.path.join(
        BASE_DIR,
        "../..",
        "data",
        "validation_datasets",
        "256_50_samples_T300_200_steps_seed0_dt5e-06",
    )

    # Standard values for isothermal model:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="isothermal_model_new")
    # make sure the dataset exists, if not use the data/data_generation/training_data.py script to generate it.
    parser.add_argument("--train_data_dir", type=str, default=default_train_dir)
    # make sure the dataset exists, if not use the data/data_generation/kmc_trajectories.py script to generate it.
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=default_val_data_path,
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_interval_percent", type=int, default=25)
    parser.add_argument("--anneal_epochs", type=int, default=20)
    parser.add_argument("--frac_sims", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=75)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dt_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00036)
    parser.add_argument("--max_negative_change", type=int, default=7)
    parser.add_argument("--max_positive_change", type=int, default=13)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--beta_kl_final", type=float, default=2.0)
    parser.add_argument("--free_bits", type=float, default=0.004)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--skip_dropout", type=float, default=0.1)
    parser.add_argument("--temp_channel", type=int, default=0)
    parser.add_argument(
        "--use_pooling",
        type=int,
        default=1,
        help="Use pooling layers to reduce spatial dimensions",
    )
    parser.add_argument("--pool_type", type=str, default="mean")
    parser.add_argument(
        "--temp_range",
        nargs="+",
        type=float,
        default=[275, 325],
        help="Min and max temperature for scaling (required if temp_channel=1)",
    )
    parser.add_argument(
        "--disable_saving", action="store_true", help="Do not save checkpoints or logs"
    )
    parser.add_argument(
        "--console_log_only", action="store_true", help="Do not write to .log files"
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Attempt to compile the model for faster training",
    )
    parser.add_argument(
        "--use_accelerate",
        action="store_true",
        help="Use Hugging Face Accelerator",
    )

    return parser.parse_args(args)


def get_vram_usage(device: torch.device) -> float:
    """
    Return peak allocated VRAM in megabytes for CUDA devices.

    Args:
        device: The target PyTorch device.

    Returns:
        Peak memory allocated in MB. Returns 0.0 if not a CUDA device.
    """
    is_cuda = torch.cuda.is_available() and device.type == "cuda"
    if is_cuda:
        return torch.cuda.max_memory_allocated(device) / BYTES_PER_MEGABYTE
    return 0.0


def _setup_run_directory(
    model_name: str, disable_saving: bool, is_main_process: bool
) -> Optional[str]:
    """
    Create a unique directory for storing model artifacts and logs.

    Args:
        model_name: The base name of the model run.
        disable_saving: Flag indicating if saving should be bypassed.
        is_main_process: Flag restricting folder creation to the primary process.

    Returns:
        The path to the created directory, or None if saving is disabled.
    """
    if disable_saving:
        return None

    models_base_dir = os.path.join(BASE_DIR, "..", "trained_models")
    run_name = model_name
    run_dir = None

    if is_main_process:
        os.makedirs(models_base_dir, exist_ok=True)

        counter = 1
        while os.path.exists(os.path.join(models_base_dir, run_name)):
            run_name = f"{model_name}_v{counter}"
            counter += 1

        run_dir = os.path.join(models_base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

    return run_dir


def _calculate_beta_kl(
    epoch: int, warmup_epochs: int, anneal_epochs: int, beta_kl_final: float
) -> float:
    """
    Calculate the current KL-divergence scaling factor (Beta) using a linear warmup schedule.

    KL annealing prevents the model from ignoring the latent space early in training
    (posterior collapse) by slowly introducing the KL penalty after a warmup period.

    Args:
        epoch: Current training epoch.
        warmup_epochs: Number of epochs to keep beta at 0.
        anneal_epochs: Number of epochs to linearly scale beta to its final value.
        beta_kl_final: Maximum value for the KL scaling factor.

    Returns:
        The scaled beta value for the current epoch.
    """
    if epoch <= warmup_epochs:
        return 0.0

    ramp_epoch = epoch - warmup_epochs
    return min(beta_kl_final, beta_kl_final * (ramp_epoch / anneal_epochs))


def _compute_cvae_loss(
    recon_profiles: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    current_beta: float,
    free_bits: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the objective function for the Conditional VAE.

    Utilizes a "free bits" formulation for the KL divergence to enforce a minimum
    information capacity in the latent space, further preventing posterior collapse.

    Args:
        recon_profiles: The predicted output logits from the decoder.
        target: The ground truth target tensor.
        mu: Latent space mean.
        logvar: Latent space log variance.
        current_beta: Current KL scaling factor.
        free_bits: Minimum KL budget per latent dimension.

    Returns:
        Tuple containing: (total_loss, reconstruction_loss, raw_kl, scaled_kl, true_loss_val)
    """
    num_classes = recon_profiles.shape[1]
    safe_target = torch.clamp(target, 0, num_classes - 1)
    rec_loss = F.cross_entropy(recon_profiles, safe_target)

    # Standard analytical KL divergence for diagonal Gaussian prior
    kl_tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_raw_per_sample = torch.sum(kl_tensor, dim=[1, 2])

    sequence_length = recon_profiles.shape[2]
    kl_raw = kl_raw_per_sample.mean() / sequence_length

    num_latent_elements = mu.shape[1] * mu.shape[2]
    min_kl_budget = free_bits * num_latent_elements

    # Apply hinge loss to ensure KL doesn't drop below the free bits threshold
    kl_hinged = torch.relu(kl_raw_per_sample - min_kl_budget).mean() / sequence_length

    loss_scaled_kl = current_beta * kl_hinged
    total_loss = rec_loss + loss_scaled_kl
    true_loss_val = rec_loss + kl_raw

    return total_loss, rec_loss, kl_raw, loss_scaled_kl, true_loss_val


def _evaluate_model(
    raw_model: torch.nn.Module, val_data_path: str, pass_temp: bool
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model against actual Kinetic Monte Carlo trajectories.

    The final validation loss is a weighted sum of the relative root mean square
    errors (RRMSE) of growth and surface roughness (rq) metrics, penalizing both
    mean deviations and variance.

    Args:
        raw_model: The core uncompiled/unwrapped PyTorch model.
        val_data_path: Path to the validation trajectory data.
        pass_temp: Whether to pass temperature conditioning to the model.

    Returns:
        Tuple containing the composite validation loss and a dictionary of individual metrics.
    """
    traj_results = trajectory_validation.trajectory_val_loss(
        kmc_data_path=val_data_path,
        ml_model=raw_model,
        return_errors_only=True,
        pass_temp_to_ml_model=pass_temp,
    )

    rrmse_growth = traj_results["nrmse_mean_growth"]
    rrmse_rq = traj_results["nrmse_mean_rq"]
    rrmse_std_growth = traj_results["nrmse_std_growth"]
    rrmse_std_rq = traj_results["nrmse_std_rq"]

    val_loss = VAL_LOSS_GROWTH_WEIGHT * (
        rrmse_growth + rrmse_rq
    ) + VAL_LOSS_STD_WEIGHT * (rrmse_std_growth + rrmse_std_rq)

    return val_loss, traj_results


def _save_checkpoint(
    model: torch.nn.Module, run_dir: Optional[str], filename: str
) -> None:
    """
    Save model weights to disk if saving is enabled.

    Args:
        model: The model whose state dict will be saved.
        run_dir: Directory to save the file, or None if saving is disabled.
        filename: Target filename for the checkpoint.
    """
    is_saving_enabled = run_dir is not None
    if is_saving_enabled:
        torch.save(model.state_dict(), os.path.join(run_dir, filename))


def run_training(
    args: argparse.Namespace,
    metrics_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
    preloaded_train_data: Optional[Dataset] = None,
) -> None:
    """
    Execute the primary training loop for the Conditional VAE model.

    Args:
        args: Parsed command-line arguments.
        metrics_callback: Optional function to stream metrics externally.
        preloaded_train_data: Optional dataset to bypass initialization.
    """
    if args.use_accelerate:
        from accelerate import Accelerator, DistributedDataParallelKwargs

        accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        num_processes = accelerator.num_processes
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        num_processes = 1

    if is_main_process:
        args_dict = vars(args)
        args_log_str = " | ".join([f"{k}: {v}" for k, v in args_dict.items()])
        print(f"Training Configuration: {args_log_str}")

    run_dir = _setup_run_directory(
        args.model_name, args.disable_saving, is_main_process
    )

    if accelerator:
        accelerator.wait_for_everyone()

    logger = setup_logger(
        args.model_name,
        log_dir=run_dir if run_dir else "",
        is_main_process=is_main_process,
        console_only=args.console_log_only or args.disable_saving,
    )

    try:
        device_name = "GPU(s)" if device.type == "cuda" else "CPU"
        logger.info(
            f"Starting training on {num_processes} {device_name}. "
            f"Saving disabled: {args.disable_saving}"
        )
        if run_dir:
            logger.info(f"All artifacts will be saved to: {run_dir}")

        has_temp_channel = bool(args.temp_channel)

        model_config = {
            "in_channels": 1,
            "max_negative_change": args.max_negative_change,
            "max_positive_change": args.max_positive_change,
            "cond_channels": 3 if has_temp_channel else 2,
            "latent_channels": args.latent_channels,
            "skip_dropout": args.skip_dropout,
            "n_layers": args.n_layers,
            "kernel_size": args.kernel_size,
            "base_channels": args.base_channels,
            "groups": MODEL_GROUPS,
            "padding_mode": "circular",
            "activation": args.activation,
            "pool_type": args.pool_type,
            "temp_channel": has_temp_channel,
            "temp_range": args.temp_range if has_temp_channel else None,
            "use_pooling": bool(args.use_pooling),
        }

        is_saving_enabled = is_main_process and (run_dir is not None)
        if is_saving_enabled:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(model_config, f, indent=4)

        model = architecture.VariationalAutoencoder_FullyConv(**model_config)

        if not accelerator:
            model = model.to(device)

        raw_model = model

        logger.info(f"Model has {model.trainable_parameters} parameters")
        logger.info(f"Model receptive field: {model.receptive_field} lattice sites")
        logger.info(
            f"Cond Channels: {model.cond_channels}, Latent Channels: {model.latent_channels}"
        )

        if metrics_callback and is_main_process:
            try:
                metrics_callback(
                    {
                        "num_parameters": float(model.trainable_parameters),
                        "receptive_field": float(getattr(model, "receptive_field", 0)),
                    },
                    step=0,
                )
            except Exception:
                logger.exception("Failed to send model metadata to metrics_callback")

        is_compilation_requested = torch.cuda.is_available() and args.compile_model
        if is_compilation_requested:
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Compilation failed: {e}. Moving on.")

        is_adamw = args.optimizer.lower() == "adamw"
        if is_adamw:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_dataset = preloaded_train_data or KMC_Single_Dataset_Relative(
            data_dir=args.train_data_dir,
            temp=has_temp_channel,
            temp_range=args.temp_range if has_temp_channel else None,
            shift=True,
            frac_sims=args.frac_sims,
            max_steps=args.max_steps,
            max_negative_change=args.max_negative_change,
            max_positive_change=args.max_positive_change,
            dt_step=args.dt_step,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=bool(args.persistent_workers),
            drop_last=True,
        )

        if accelerator:
            model, optimizer, train_loader = accelerator.prepare(
                model, optimizer, train_loader
            )

        best_val_loss = float("inf")
        best_val_epoch = -1
        global_step = 0
        training_history = {}

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            is_cuda = torch.cuda.is_available() and device.type == "cuda"
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            model.train()
            (
                train_loss,
                train_recon,
                train_kl_raw,
                train_kl_scaled,
                train_true_loss,
            ) = (0.0, 0.0, 0.0, 0.0, 0.0)

            total_batches = len(train_loader)
            log_every_n_batches = max(
                1, int(total_batches * (args.log_interval_percent / 100.0))
            )

            current_beta = _calculate_beta_kl(
                epoch, args.warmup_epochs, args.anneal_epochs, args.beta_kl_final
            )

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1
                optimizer.zero_grad()
                start_profiles, model_input, target = [
                    b.to(device, non_blocking=True) for b in batch
                ]

                if accelerator:
                    with accelerator.autocast():
                        recon_profiles, mu, logvar = model(
                            x=model_input,
                            cond=start_profiles,
                        )
                        if torch.isnan(mu).any():
                            logger.error(
                                "Latent space collapsed to NaN! Reduce Learning Rate."
                            )

                        (
                            loss,
                            rec_loss,
                            kl_raw,
                            loss_scaled_kl,
                            true_loss_val,
                        ) = _compute_cvae_loss(
                            recon_profiles,
                            target,
                            mu,
                            logvar,
                            current_beta,
                            args.free_bits,
                        )

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                else:
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if is_cuda
                        else torch.autocast(device_type="cpu")
                    ):
                        recon_profiles, mu, logvar = model(
                            x=model_input,
                            cond=start_profiles,
                        )
                        if torch.isnan(mu).any():
                            logger.error(
                                "Latent space collapsed to NaN! Reduce Learning Rate."
                            )

                        (
                            loss,
                            rec_loss,
                            kl_raw,
                            loss_scaled_kl,
                            true_loss_val,
                        ) = _compute_cvae_loss(
                            recon_profiles,
                            target,
                            mu,
                            logvar,
                            current_beta,
                            args.free_bits,
                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()
                train_recon += rec_loss.item()
                train_kl_raw += kl_raw.item()
                train_kl_scaled += loss_scaled_kl.item()
                train_true_loss += true_loss_val.item()

                current_step = batch_idx + 1
                is_logging_step = (
                    current_step % log_every_n_batches == 0
                    or current_step == total_batches
                )

                if is_logging_step:
                    elapsed_epoch_time = time.time() - epoch_start_time
                    time_per_batch = elapsed_epoch_time / current_step
                    remaining_batches = total_batches - current_step
                    epoch_eta_seconds = remaining_batches * time_per_batch
                    epoch_eta_str = str(
                        datetime.timedelta(seconds=int(epoch_eta_seconds))
                    )

                    vram_mb = get_vram_usage(device)

                    step_train_metrics = {
                        "train_loss": train_loss / current_step,
                        "train_true_loss": train_true_loss / current_step,
                        "train_recon_loss": train_recon / current_step,
                        "train_kl_raw": train_kl_raw / current_step,
                        "train_kl_scaled": train_kl_scaled / current_step,
                        "vram_mb": vram_mb,
                    }

                    metrics_str = " | ".join(
                        [
                            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in step_train_metrics.items()
                        ]
                    )
                    logger.info(
                        f"E{epoch} [{current_step}/{total_batches}] | Epoch ETA: {epoch_eta_str} | Beta: {current_beta:.3f} | {metrics_str}"
                    )

                    if metrics_callback and is_main_process:
                        step_train_metrics["epoch"] = epoch
                        step_train_metrics["beta_kl"] = current_beta
                        metrics_callback(step_train_metrics, step=global_step)

            epoch_train_metrics = {
                "train_loss": train_loss / total_batches,
                "train_true_loss": train_true_loss / total_batches,
                "train_recon_loss": train_recon / total_batches,
                "train_kl_raw": train_kl_raw / total_batches,
                "train_kl_scaled": train_kl_scaled / total_batches,
                "peak_vram_mb": get_vram_usage(device),
            }

            val_start_time = time.time()
            model.eval()

            did_validate = False
            val_metrics_log = {}

            is_validation_epoch = epoch == 1 or epoch >= (
                args.warmup_epochs + args.anneal_epochs
            )

            if is_validation_epoch:
                did_validate = True
                val_path = os.path.join(BASE_DIR, args.val_data_path)
                val_loss, traj_results = _evaluate_model(
                    raw_model, val_path, has_temp_channel
                )

                val_metrics_log = {
                    "validation_loss": val_loss,
                    **traj_results,
                }

            val_duration = time.time() - val_start_time

            if did_validate:
                val_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in val_metrics_log.items()]
                )
                logger.info(
                    f"Epoch {epoch} Validation | Val Time: {val_duration:.1f}s | {val_str}"
                )

                if metrics_callback and is_main_process:
                    val_metrics_log["epoch"] = epoch
                    metrics_callback(val_metrics_log, step=global_step)
            else:
                logger.info(f"Epoch {epoch} Validation | Skipped during annealing.")
                val_loss = float("inf")

            if is_main_process:
                epoch_history = {
                    "epoch": epoch,
                    "beta_kl": current_beta,
                    "epoch_train_time_seconds": time.time()
                    - epoch_start_time
                    - val_duration,
                    **epoch_train_metrics,
                    **val_metrics_log,
                }

                for k, v in epoch_history.items():
                    if k not in training_history:
                        training_history[k] = []
                    training_history[k].append(v)

                if is_saving_enabled:
                    history_path = os.path.join(run_dir, "history.json")
                    with open(history_path, "w") as f:
                        json.dump(training_history, f, indent=4)

                _save_checkpoint(raw_model, run_dir, "last_model.pt")

                is_new_best_model = did_validate and val_loss < best_val_loss
                if is_new_best_model:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    _save_checkpoint(raw_model, run_dir, "best_model.pt")
                    logger.info(f" -> New best model saved (Val Loss: {val_loss:.4f})")
                elif did_validate:
                    logger.info(
                        f" -> No improvement over best model (Best Val Loss: {best_val_loss:.4f}, saved at epoch {best_val_epoch})"
                    )

        logger.info("Training finished successfully.")

    except Exception:
        logger.exception("Fatal error:")
        raise


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
