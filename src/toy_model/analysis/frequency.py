"""Frequency-domain analysis for one-dimensional signals."""

from __future__ import annotations

import math

import torch


def compute_amplitude_spectrum(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    remove_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the one-sided amplitude spectrum of a sampled signal.

    Args:
        x:
            Uniformly spaced coordinates with shape [N] or [N, 1].
        y:
            Signal values with shape [N] or [N, 1].
        remove_mean:
            Whether to remove the signal mean before computing the FFT.

    Returns:
        A tuple ``(angular_frequencies, amplitudes)``.

        Angular frequencies are measured in radians per x-unit. Therefore,
        ``sin(x)`` should have a peak near angular frequency 1, and
        ``sin(10x)`` should have a peak near angular frequency 10.
    """
    x_flat = x.detach().cpu().reshape(-1)
    y_flat = y.detach().cpu().reshape(-1)

    if x_flat.numel() != y_flat.numel():
        raise ValueError("x and y must contain the same number of values.")

    if x_flat.numel() < 2:
        raise ValueError("At least two samples are required.")

    differences = x_flat[1:] - x_flat[:-1]
    spacing = float(differences.mean().item())

    if spacing <= 0:
        raise ValueError("x must be strictly increasing.")

    expected_differences = torch.full_like(
        differences,
        spacing,
    )

    if not torch.allclose(
        differences,
        expected_differences,
        rtol=1e-4,
        atol=1e-7,
    ):
        raise ValueError("x must be uniformly spaced.")

    if remove_mean:
        y_flat = y_flat - y_flat.mean()

    num_samples = y_flat.numel()

    coefficients = torch.fft.rfft(y_flat)

    amplitudes = (
        2.0
        * coefficients.abs()
        / num_samples
    )

    # The DC component must not be doubled.
    amplitudes[0] *= 0.5

    # For an even number of samples, the Nyquist component must not be
    # doubled either.
    if num_samples % 2 == 0:
        amplitudes[-1] *= 0.5

    cycle_frequencies = torch.fft.rfftfreq(
        num_samples,
        d=spacing,
    )

    angular_frequencies = (
        2.0
        * math.pi
        * cycle_frequencies
    )

    return angular_frequencies, amplitudes


def find_nearest_frequency_amplitude(
    angular_frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    target_frequency: float,
) -> float:
    """Return the amplitude at the nearest available frequency bin."""
    if target_frequency < 0:
        raise ValueError("target_frequency cannot be negative.")

    index = torch.argmin(
        torch.abs(
            angular_frequencies
            - target_frequency
        )
    )

    return float(amplitudes[index].item())
