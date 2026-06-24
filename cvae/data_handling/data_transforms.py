from typing import Optional, Tuple

import torch


def scale_inv_temperature_to_minus1_1(
    temperature_K: torch.Tensor, temp_range: Tuple[float, float]
) -> torch.Tensor:
    """
    Scale temperature into the interval [-1, 1] using an inverse relationship.

    In Kinetic Monte Carlo, transition rates scale exponentially with inverse temperature
    (Arrhenius behavior: exp(-E / kT)). Feeding the neural network 1/T rather than T
    provides a much more linear and learnable feature representation for the physical kinetics.

    Args:
        temperature_K: The current simulation temperature in Kelvin.
        temp_range: Tuple of (min_temperature, max_temperature) defining the scaling bounds.

    Returns:
        A scalar tensor containing the normalized temperature value.
    """
    min_temp, max_temp = temp_range

    inv_temp = 1.0 / temperature_K
    min_inv_temp = 1.0 / max_temp
    max_inv_temp = 1.0 / min_temp

    # Standard Min-Max scaling to [0, 1]
    scaled_0_1 = (inv_temp - min_inv_temp) / (max_inv_temp - min_inv_temp)

    # Shift to [-1, 1] and invert so higher real temperatures map to higher scaled values
    return (scaled_0_1 * 2.0 - 1.0) * -1.0


def _create_temperature_channel(
    start_lattice: torch.Tensor,
    temperature_kelvin: float,
    temp_range: Tuple[float, float],
) -> torch.Tensor:
    """
    Create a spatial temperature channel by broadcasting the scaled scalar temperature.

    Args:
        start_lattice: The base lattice tensor used to determine the target spatial dimensions.
        temperature_kelvin: The absolute temperature to scale and broadcast.
        temp_range: The globally defined min/max temperature bounds.

    Returns:
        A tensor containing the broadcasted temperature channel.
    """
    temp_scaled = scale_inv_temperature_to_minus1_1(
        torch.tensor(temperature_kelvin), temp_range
    )

    temp_channel = torch.full_like(
        start_lattice, fill_value=temp_scaled.item(), dtype=torch.float
    )

    return temp_channel


def create_input_condition(
    start_lattice: torch.Tensor,
    max_negative_change: int,
    max_positive_change: int,
    temp: bool = False,
    temperature_kelvin: Optional[float] = None,
    temp_range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Construct the multi-channel spatial input tensor for the neural network.

    Converts the raw lattice heights into local relative features (slopes) and boundary
    conditions (masks) to ensure the model learns generalized local interactions rather
    than memorizing absolute global heights.

    Args:
        start_lattice: The raw 1D surface height profile.
        max_negative_change: Maximum allowed downward height change.
        max_positive_change: Maximum allowed upward height change.
        temp: Flag indicating whether to append a temperature condition channel.
        temperature_kelvin: The absolute temperature (required if temp=True).
        temp_range: The min/max temperature bounds (required if temp=True).

    Returns:
        A tuple containing the concatenated multi-channel input tensor and the scaled temperature scalar.
    """
    max_change = max_negative_change + max_positive_change

    # Ensure the tensor has a channel dimension [Batch, Channel, Spatial_Length]
    if start_lattice.dim() == 2:
        start_lattice = start_lattice.unsqueeze(1)

    # Calculate the local gradient (slope) between adjacent sites.
    # This is the primary driver for local KMC step probabilities.
    diff = torch.roll(start_lattice, shifts=-1, dims=2) - start_lattice
    slope_channel = torch.clamp(diff, -max_change, max_change) / max_change

    # Create a mask representing how much material is actually available to be removed.
    # Prevents the model from predicting desorption/etching below the lattice floor (height = 0).
    mask_channel = (
        torch.clamp(start_lattice, 0, max_negative_change) / max_negative_change
    )

    channels_to_stack = [slope_channel, mask_channel]

    if temp:
        if temperature_kelvin is None or temp_range is None:
            raise ValueError(
                "temperature_kelvin and temp_range must be provided if temp=True"
            )

        temp_channel = _create_temperature_channel(
            start_lattice, temperature_kelvin, temp_range
        )
        channels_to_stack.append(temp_channel)

    stacked_input = torch.cat(channels_to_stack, dim=1)

    return stacked_input


def create_targets(
    start_lattice: torch.Tensor,
    final_lattice: torch.Tensor,
    max_negative_change: int,
    max_positive_change: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the normalized regression target and discrete classification target for the lattice evolution.

    Args:
        start_lattice: The initial surface height profile.
        final_lattice: The surface height profile after `dt_step` KMC steps.
        max_negative_change: The lower bound for height difference clamping.
        max_positive_change: The upper bound for height difference clamping.

    Returns:
        A tuple containing the normalized continuous delta and the discrete integer class targets.
    """
    delta = final_lattice - start_lattice
    max_abs_change = max(max_negative_change, max_positive_change)

    delta_clamped = torch.clamp(delta, -max_negative_change, max_positive_change)

    # Normalized delta primarily used for VAE reconstruction loss
    delta_normalized = delta_clamped / max_abs_change

    # Shift the clamped delta by `max_negative_change` to ensure all class indices are >= 0.
    # This is strictly required for PyTorch's CrossEntropyLoss which expects class indices in [0, C-1].
    target_shifted = delta_clamped + max_negative_change

    num_classes = max_negative_change + max_positive_change + 1
    target_classes = torch.clamp(target_shifted.squeeze(0), 0, num_classes).long()

    return delta_normalized, target_classes
