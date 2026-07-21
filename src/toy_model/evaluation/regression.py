"""Evaluation utilities for one-dimensional regression experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from toy_model.data import evaluate_function
from toy_model.training import predict


@dataclass(frozen=True)
class FunctionRegressionMetrics:
    """Metrics separating training fit, interpolation and extrapolation."""

    train_label_mse: float
    train_clean_mse: float
    interpolation_mse: float
    extrapolation_mse: float
    interpolation_max_abs_error: float
    extrapolation_max_abs_error: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics into a serializable dictionary."""
        return asdict(self)


def _mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(
        torch.mean(
            (prediction - target) ** 2
        ).item()
    )


def _masked_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if not bool(mask.any()):
        return float("nan")

    squared_error = (
        prediction.squeeze(1)
        - target.squeeze(1)
    ) ** 2

    return float(
        squared_error[mask].mean().item()
    )


def _masked_max_abs_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if not bool(mask.any()):
        return float("nan")

    absolute_error = torch.abs(
        prediction.squeeze(1)
        - target.squeeze(1)
    )

    return float(
        absolute_error[mask].max().item()
    )


@torch.no_grad()
def evaluate_function_regression(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    function_name: str,
    train_min: float,
    train_max: float,
    evaluation_min: float,
    evaluation_max: float,
    grid_points: int = 4096,
) -> tuple[
    FunctionRegressionMetrics,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate a model against noisy labels and the clean target function.

    Returns:
        A tuple containing:

        - metrics
        - evaluation coordinates
        - clean ground-truth values
        - model predictions
    """
    if train_min >= train_max:
        raise ValueError("train_min must be smaller than train_max.")

    if evaluation_min >= evaluation_max:
        raise ValueError(
            "evaluation_min must be smaller than evaluation_max."
        )

    if evaluation_min > train_min or evaluation_max < train_max:
        raise ValueError(
            "The evaluation interval must contain the training interval."
        )

    if grid_points < 2:
        raise ValueError("grid_points must be at least two.")

    x_train_cpu = x_train.detach().cpu()
    y_train_cpu = y_train.detach().cpu()

    train_prediction = predict(
        model=model,
        x=x_train_cpu,
    )

    clean_train_target = evaluate_function(
        x=x_train_cpu,
        function_name=function_name,
    )

    x_grid = torch.linspace(
        evaluation_min,
        evaluation_max,
        grid_points,
    ).unsqueeze(1)

    clean_grid_target = evaluate_function(
        x=x_grid,
        function_name=function_name,
    )

    grid_prediction = predict(
        model=model,
        x=x_grid,
    )

    x_values = x_grid.squeeze(1)

    interpolation_mask = (
        (x_values >= train_min)
        & (x_values <= train_max)
    )

    extrapolation_mask = ~interpolation_mask

    metrics = FunctionRegressionMetrics(
        train_label_mse=_mean_squared_error(
            train_prediction,
            y_train_cpu,
        ),
        train_clean_mse=_mean_squared_error(
            train_prediction,
            clean_train_target,
        ),
        interpolation_mse=_masked_mean_squared_error(
            grid_prediction,
            clean_grid_target,
            interpolation_mask,
        ),
        extrapolation_mse=_masked_mean_squared_error(
            grid_prediction,
            clean_grid_target,
            extrapolation_mask,
        ),
        interpolation_max_abs_error=_masked_max_abs_error(
            grid_prediction,
            clean_grid_target,
            interpolation_mask,
        ),
        extrapolation_max_abs_error=_masked_max_abs_error(
            grid_prediction,
            clean_grid_target,
            extrapolation_mask,
        ),
    )

    return (
        metrics,
        x_grid,
        clean_grid_target,
        grid_prediction,
    )
