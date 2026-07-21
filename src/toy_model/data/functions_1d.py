"""Synthetic one-dimensional functions and sampling utilities."""

from __future__ import annotations

import math

import torch


def evaluate_function(
    x: torch.Tensor,
    function_name: str,
) -> torch.Tensor:
    """Evaluate a synthetic one-dimensional target function.

    Args:
        x:
            Input tensor. The expected shape is [num_samples, 1].
        function_name:
            Name of the target function.

    Returns:
        Function values with the same shape as ``x``.
    """
    if function_name == "linear":
        return 2.0 * x + 1.0

    if function_name == "sine":
        return torch.sin(x)

    if function_name == "high_frequency_sine":
        return torch.sin(10.0 * x)

    if function_name == "mixed_sine":
        return torch.sin(x) + 0.3 * torch.sin(10.0 * x)

    raise ValueError(f"Unknown function: {function_name!r}")


def sample_function(
    function_name: str,
    num_samples: int,
    x_min: float = -math.pi,
    x_max: float = math.pi,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a synthetic regression dataset.

    Args:
        function_name:
            Name passed to ``evaluate_function``.
        num_samples:
            Number of examples to generate.
        x_min:
            Lower sampling boundary.
        x_max:
            Upper sampling boundary.
        noise_std:
            Standard deviation of additive Gaussian label noise.
        seed:
            Optional seed used only for this sampling operation.

    Returns:
        A tuple ``(x, y)`` where both tensors have shape
        ``[num_samples, 1]``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")

    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")

    generator = torch.Generator()

    if seed is not None:
        generator.manual_seed(seed)

    x = torch.empty(num_samples, 1).uniform_(
        x_min,
        x_max,
        generator=generator,
    )

    y = evaluate_function(x, function_name)

    if noise_std > 0:
        noise = torch.randn(
            y.shape,
            generator=generator,
            dtype=y.dtype,
        )
        y = y + noise_std * noise

    return x, y

def sample_function_grid(
    function_name: str,
    num_samples: int,
    x_min: float = -math.pi,
    x_max: float = math.pi,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a function on an evenly spaced one-dimensional grid.

    Unlike ``sample_function``, this function uses ``torch.linspace``.
    It is therefore suitable for experiments that explicitly compare
    sampling density.

    Args:
        function_name:
            Name passed to ``evaluate_function``.
        num_samples:
            Number of grid points, including both interval endpoints.
        x_min:
            Lower endpoint.
        x_max:
            Upper endpoint.
        noise_std:
            Standard deviation of optional Gaussian label noise.
        seed:
            Optional seed for label noise.

    Returns:
        A tuple ``(x, y)`` with shape [num_samples, 1].
    """
    if num_samples < 2:
        raise ValueError("num_samples must be at least two.")

    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")

    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")

    x = torch.linspace(
        x_min,
        x_max,
        num_samples,
    ).unsqueeze(1)

    y = evaluate_function(
        x=x,
        function_name=function_name,
    )

    if noise_std > 0:
        if seed is None:
            noise = torch.randn_like(y)
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)

            noise = torch.randn(
                y.shape,
                generator=generator,
                dtype=y.dtype,
            )

        y = y + noise_std * noise

    return x, y
