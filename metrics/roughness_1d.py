import functools
from typing import Union

import numpy as np
import torch

TensorOrArray = Union[np.ndarray, torch.Tensor]


def _preserve_tensor_type(func):
    """
    Decorator to abstract away PyTorch to NumPy conversion boilerplate.
    Transforms incoming Tensors to NumPy arrays for computation, and
    wraps the results back into Tensors on the original device.
    """

    @functools.wraps(func)
    def wrapper(heights: TensorOrArray, *args, **kwargs) -> TensorOrArray:
        is_tensor = isinstance(heights, torch.Tensor)
        if is_tensor:
            device = heights.device
            dtype = heights.dtype
            h_np = heights.detach().cpu().numpy()
        else:
            h_np = heights

        result = func(h_np, *args, **kwargs)

        if is_tensor:
            # count_dendrites returns counts, which require integer/long types.
            # All other metric calculations preserve the original input float dtype.
            out_dtype = torch.long if np.issubdtype(result.dtype, np.integer) else dtype
            return torch.from_numpy(result).to(device=device, dtype=out_dtype)

        return result

    return wrapper


@_preserve_tensor_type
def Ra(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate Average Roughness (Ra) as the mean absolute deviation from the mean line.

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    # Using axis=-1 targets the final spatial dimension (L) universally,
    # eliminating the need for hardcoded 1D, 2D, or 3D conditional logic.
    mean_height = np.mean(heights, axis=-1, keepdims=True)
    return np.mean(np.abs(heights - mean_height), axis=-1)


@_preserve_tensor_type
def Rq(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate Root Mean Square Roughness (Rq or Rms).

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    mean_height = np.mean(heights, axis=-1, keepdims=True)
    return np.sqrt(np.mean(np.square(heights - mean_height), axis=-1))


@_preserve_tensor_type
def Rv(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate Maximum Valley Depth (Rv) below the mean line.

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    mean_height = np.mean(heights, axis=-1, keepdims=True)
    return np.abs(np.min(heights - mean_height, axis=-1))


@_preserve_tensor_type
def Rp(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate Maximum Peak Height (Rp) above the mean line.

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    mean_height = np.mean(heights, axis=-1, keepdims=True)
    return np.max(heights - mean_height, axis=-1)


@_preserve_tensor_type
def Rz(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate Maximum Peak to Valley Height (Rz).

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    return np.max(heights, axis=-1) - np.min(heights, axis=-1)


@_preserve_tensor_type
def count_dendrites(heights: TensorOrArray, min_length: int = 1) -> TensorOrArray:
    """
    Count the number of contiguous non-zero regions (dendrites) in a profile.

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].
        min_length: Minimum contiguous sites required to be considered a valid dendrite.

    Returns:
        Counts of shape [], [B], or [B, C] with integer types.
    """
    spatial_length = heights.shape[-1]
    if spatial_length < min_length:
        return np.zeros(heights.shape[:-1], dtype=np.int64)

    mask = heights != 0
    valid_seq_mask = mask.copy()

    # Identify sequences of truth values that meet the minimum length threshold
    for i in range(1, min_length):
        valid_seq_mask &= np.roll(mask, shift=-i, axis=-1)

    # Use a bitwise difference with a right-shifted mask to detect rising edges
    # (the start of a new dendrite), allowing us to count occurrences with `.sum()`.
    shifted_mask = np.roll(valid_seq_mask, shift=1, axis=-1)
    dendrite_starts = valid_seq_mask & ~shifted_mask

    counts = dendrite_starts.sum(axis=-1)

    all_filled = mask.all(axis=-1)
    return counts + all_filled.astype(np.int64)


@_preserve_tensor_type
def occupation_per_height(
    heights: TensorOrArray, cumulative: bool = True
) -> TensorOrArray:
    """
    Calculate the mean occupation/density profile across structural height levels.

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].
        cumulative: If True, calculates the density profile. If False, calculates the exact Height PDF.

    Returns:
        Density array of shape [..., H_max + 1].
    """
    max_h = int(np.max(heights))
    levels = np.arange(max_h + 1)

    if cumulative:
        occupations = heights[..., np.newaxis] >= levels
    else:
        occupations = heights[..., np.newaxis] == levels

    return np.mean(occupations, axis=-2)


@_preserve_tensor_type
def thickness(heights: TensorOrArray) -> TensorOrArray:
    """
    Calculate the absolute thickness of the profile (maximum absolute height).

    Args:
        heights: Surface profiles of shape [L], [B, L], or [B, C, L].

    Returns:
        Scalar, [B], or [B, C] shape corresponding to the input dimensionality.
    """
    return np.max(heights, axis=-1)
