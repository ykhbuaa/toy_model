"""Diagnostics for controlled interpolation and extrapolation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from toy_model.training import predict


@dataclass(frozen=True)
class SymmetryMetrics:
    """Relative odd-symmetry and even-symmetry residuals."""

    odd_residual_relative: float
    even_residual_relative: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AffineFitMetrics:
    """Metrics obtained by fitting y = slope * x + intercept."""

    slope: float
    intercept: float
    r_squared: float

    def to_dict(
        self,
        prefix: str,
    ) -> dict[str, float]:
        return {
            f"{prefix}_slope": self.slope,
            f"{prefix}_intercept": self.intercept,
            f"{prefix}_r_squared": self.r_squared,
        }


@dataclass(frozen=True)
class BoundaryDistanceMetric:
    """Error in one distance bin outside the training support."""

    distance_min: float
    distance_max: float
    mse: float
    num_points: int


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute MSE on a Boolean mask."""
    prediction_flat = (
        prediction
        .detach()
        .cpu()
        .reshape(-1)
    )

    target_flat = (
        target
        .detach()
        .cpu()
        .reshape(-1)
    )

    mask_flat = (
        mask
        .detach()
        .cpu()
        .reshape(-1)
    )

    if not bool(mask_flat.any()):
        return float("nan")

    squared_error = (
        prediction_flat[mask_flat]
        - target_flat[mask_flat]
    ) ** 2

    return float(
        squared_error.mean().item()
    )


def compute_fixed_zone_metrics(
    x: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    zone_boundaries: tuple[float, float, float],
) -> dict[str, float]:
    """Compute MSE on three fixed absolute evaluation zones.

    The zones are independent of each model's training interval:

        Zone 0: |x| <= boundary_0
        Zone 1: boundary_0 < |x| <= boundary_1
        Zone 2: boundary_1 < |x| <= boundary_2

    This makes metrics from different training intervals directly
    comparable.
    """
    boundary_0, boundary_1, boundary_2 = zone_boundaries

    if not 0 < boundary_0 < boundary_1 < boundary_2:
        raise ValueError(
            "zone_boundaries must be strictly increasing and positive."
        )

    absolute_x = (
        x
        .detach()
        .cpu()
        .reshape(-1)
        .abs()
    )

    zone_0_mask = absolute_x <= boundary_0

    zone_1_mask = (
        (absolute_x > boundary_0)
        & (absolute_x <= boundary_1)
    )

    zone_2_mask = (
        (absolute_x > boundary_1)
        & (absolute_x <= boundary_2)
    )

    return {
        "zone_0_mse": _masked_mse(
            prediction,
            target,
            zone_0_mask,
        ),
        "zone_1_mse": _masked_mse(
            prediction,
            target,
            zone_1_mask,
        ),
        "zone_2_mse": _masked_mse(
            prediction,
            target,
            zone_2_mask,
        ),
    }


def compute_support_metrics(
    x: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    train_radius: float,
) -> dict[str, float]:
    """Compute error inside and outside the model's own support.

    These metrics are useful for describing one model, but the
    ``outside_support_mse`` values should not be compared blindly across
    different train radii because their integration regions differ.
    """
    if train_radius <= 0:
        raise ValueError("train_radius must be positive.")

    absolute_x = (
        x
        .detach()
        .cpu()
        .reshape(-1)
        .abs()
    )

    support_mask = absolute_x <= train_radius
    outside_mask = absolute_x > train_radius

    return {
        "support_mse": _masked_mse(
            prediction,
            target,
            support_mask,
        ),
        "outside_support_mse": _masked_mse(
            prediction,
            target,
            outside_mask,
        ),
    }


@torch.no_grad()
def compute_symmetry_metrics(
    model: nn.Module,
    *,
    max_abs_x: float,
    num_points: int = 2048,
    epsilon: float = 1e-12,
) -> SymmetryMetrics:
    """Measure how close a scalar model is to odd or even symmetry.

    Odd residual:

        E[(f(x) + f(-x))^2]
        --------------------
        E[f(x)^2 + f(-x)^2]

    Even residual:

        E[(f(x) - f(-x))^2]
        --------------------
        E[f(x)^2 + f(-x)^2]
    """
    if max_abs_x <= 0:
        raise ValueError("max_abs_x must be positive.")

    if num_points < 2:
        raise ValueError(
            "num_points must be at least two."
        )

    positive_x = torch.linspace(
        0.0,
        max_abs_x,
        num_points,
    ).unsqueeze(1)

    negative_x = -positive_x

    positive_prediction = predict(
        model=model,
        x=positive_x,
    )

    negative_prediction = predict(
        model=model,
        x=negative_x,
    )

    denominator = torch.mean(
        positive_prediction.square()
        + negative_prediction.square()
    )

    odd_numerator = torch.mean(
        (
            positive_prediction
            + negative_prediction
        ).square()
    )

    even_numerator = torch.mean(
        (
            positive_prediction
            - negative_prediction
        ).square()
    )

    denominator_value = (
        float(denominator.item())
        + epsilon
    )

    return SymmetryMetrics(
        odd_residual_relative=(
            float(odd_numerator.item())
            / denominator_value
        ),
        even_residual_relative=(
            float(even_numerator.item())
            / denominator_value
        ),
    )


@torch.no_grad()
def compute_periodicity_residual(
    model: nn.Module,
    *,
    period: float,
    x_min: float,
    x_max: float,
    num_points: int = 2048,
    epsilon: float = 1e-12,
) -> float:
    """Measure relative disagreement between f(x) and f(x + period)."""
    if period <= 0:
        raise ValueError("period must be positive.")

    if x_min >= x_max:
        raise ValueError(
            "x_min must be smaller than x_max."
        )

    if num_points < 2:
        raise ValueError(
            "num_points must be at least two."
        )

    x = torch.linspace(
        x_min,
        x_max,
        num_points,
    ).unsqueeze(1)

    prediction = predict(
        model=model,
        x=x,
    )

    shifted_prediction = predict(
        model=model,
        x=x + period,
    )

    numerator = torch.mean(
        (
            prediction
            - shifted_prediction
        ).square()
    )

    denominator = torch.mean(
        prediction.square()
        + shifted_prediction.square()
    )

    return float(
        numerator.item()
        / (
            float(denominator.item())
            + epsilon
        )
    )


def fit_affine_function(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> AffineFitMetrics:
    """Fit y = slope * x + intercept using least squares."""
    x_flat = (
        x
        .detach()
        .cpu()
        .reshape(-1)
        .to(torch.float64)
    )

    y_flat = (
        y
        .detach()
        .cpu()
        .reshape(-1)
        .to(torch.float64)
    )

    if x_flat.numel() != y_flat.numel():
        raise ValueError(
            "x and y must contain the same number of values."
        )

    if x_flat.numel() < 2:
        raise ValueError(
            "At least two points are required."
        )

    design_matrix = torch.stack(
        [
            x_flat,
            torch.ones_like(x_flat),
        ],
        dim=1,
    )

    solution = torch.linalg.lstsq(
        design_matrix,
        y_flat.unsqueeze(1),
    ).solution.squeeze(1)

    slope = solution[0]
    intercept = solution[1]

    fitted = (
        slope * x_flat
        + intercept
    )

    residual_sum_squares = torch.sum(
        (
            y_flat
            - fitted
        ).square()
    )

    total_sum_squares = torch.sum(
        (
            y_flat
            - y_flat.mean()
        ).square()
    )

    if float(total_sum_squares.item()) <= epsilon:
        r_squared = (
            1.0
            if float(residual_sum_squares.item()) <= epsilon
            else 0.0
        )
    else:
        r_squared = 1.0 - float(
            residual_sum_squares.item()
            / total_sum_squares.item()
        )

    return AffineFitMetrics(
        slope=float(slope.item()),
        intercept=float(intercept.item()),
        r_squared=r_squared,
    )


def compute_tail_affine_metrics(
    x: torch.Tensor,
    prediction: torch.Tensor,
    *,
    inner_abs_x: float,
    outer_abs_x: float,
) -> tuple[
    AffineFitMetrics,
    AffineFitMetrics,
]:
    """Fit independent affine functions to left and right tails."""
    if not 0 < inner_abs_x < outer_abs_x:
        raise ValueError(
            "Require 0 < inner_abs_x < outer_abs_x."
        )

    x_flat = (
        x
        .detach()
        .cpu()
        .reshape(-1)
    )

    prediction_flat = (
        prediction
        .detach()
        .cpu()
        .reshape(-1)
    )

    left_mask = (
        (x_flat >= -outer_abs_x)
        & (x_flat <= -inner_abs_x)
    )

    right_mask = (
        (x_flat >= inner_abs_x)
        & (x_flat <= outer_abs_x)
    )

    left_metrics = fit_affine_function(
        x=x_flat[left_mask],
        y=prediction_flat[left_mask],
    )

    right_metrics = fit_affine_function(
        x=x_flat[right_mask],
        y=prediction_flat[right_mask],
    )

    return left_metrics, right_metrics


def compute_boundary_distance_metrics(
    x: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    train_radius: float,
    distance_edges: tuple[float, ...],
) -> list[BoundaryDistanceMetric]:
    """Evaluate error using distance from the training boundary.

    For a symmetric training support [-R, R], define:

        d = |x| - R

    Only points with d >= 0 are outside the training support. Evaluating
    models at equal d makes comparisons between different R values fairer.
    """
    if train_radius <= 0:
        raise ValueError(
            "train_radius must be positive."
        )

    if len(distance_edges) < 2:
        raise ValueError(
            "distance_edges must contain at least two values."
        )

    if distance_edges[0] != 0.0:
        raise ValueError(
            "distance_edges must start at zero."
        )

    if any(
        left >= right
        for left, right in zip(
            distance_edges[:-1],
            distance_edges[1:],
        )
    ):
        raise ValueError(
            "distance_edges must be strictly increasing."
        )

    absolute_x = (
        x
        .detach()
        .cpu()
        .reshape(-1)
        .abs()
    )

    distance = absolute_x - train_radius

    metrics: list[BoundaryDistanceMetric] = []

    for index, (
        distance_min,
        distance_max,
    ) in enumerate(
        zip(
            distance_edges[:-1],
            distance_edges[1:],
        )
    ):
        is_last_bin = (
            index
            == len(distance_edges) - 2
        )

        if is_last_bin:
            mask = (
                (distance >= distance_min)
                & (distance <= distance_max)
            )
        else:
            mask = (
                (distance >= distance_min)
                & (distance < distance_max)
            )

        metrics.append(
            BoundaryDistanceMetric(
                distance_min=distance_min,
                distance_max=distance_max,
                mse=_masked_mse(
                    prediction=prediction,
                    target=target,
                    mask=mask,
                ),
                num_points=int(
                    mask.sum().item()
                ),
            )
        )

    return metrics
