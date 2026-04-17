"""
Time‑series augmentations for contrastive learning.
"""
import numpy as np
import torch


def add_gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise to the time series."""
    noise = torch.randn_like(x) * std
    return x + noise


def scale_magnitude(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Randomly scale the magnitude of the time series."""
    scale = np.random.uniform(*scale_range)
    return x * scale


def time_shift(x: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
    """Randomly shift the time series along the time axis."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        return torch.cat([x[shift:], x[:shift]], dim=0)
    elif shift < 0:
        shift = abs(shift)
        return torch.cat([x[-shift:], x[:-shift]], dim=0)
    return x


def dropout_regions(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Randomly drop out contiguous regions of the time series."""
    mask = torch.ones_like(x)
    for i in range(x.shape[0]):
        if np.random.random() < drop_prob:
            drop_len = np.random.randint(1, max(2, x.shape[0] // 10))
            start = np.random.randint(0, x.shape[0] - drop_len)
            mask[i, start:start+drop_len] = 0
    return x * mask


def apply_augmentations(x: torch.Tensor, strength: str = "medium") -> torch.Tensor:
    """
    Apply a stochastic composition of augmentations.
    Strength can be 'light', 'medium', or 'strong'.
    """
    if strength == "light":
        std, scale_range, max_shift, drop_prob = 0.02, (0.9, 1.1), 2, 0.05
    elif strength == "strong":
        std, scale_range, max_shift, drop_prob = 0.10, (0.6, 1.4), 10, 0.2
    else:  # medium
        std, scale_range, max_shift, drop_prob = 0.05, (0.8, 1.2), 5, 0.1

    x_aug = add_gaussian_noise(x, std=std)
    if np.random.random() < 0.5:
        x_aug = scale_magnitude(x_aug, scale_range=scale_range)
    if np.random.random() < 0.5:
        x_aug = time_shift(x_aug, max_shift=max_shift)
    if np.random.random() < 0.5:
        x_aug = dropout_regions(x_aug, drop_prob=drop_prob)

    return x_aug
