import math

import torch
from torch import nn

from toy_model.data import sample_function_grid
from toy_model.evaluation import (
    compute_fixed_zone_metrics,
    compute_symmetry_metrics,
    fit_affine_function,
)
from toy_model.models import FourierFeatureEncoding


def test_fixed_density_doubles_interval_with_511_points() -> None:
    x_narrow, _ = sample_function_grid(
        function_name="sine",
        num_samples=256,
        x_min=-math.pi,
        x_max=math.pi,
    )

    x_wide, _ = sample_function_grid(
        function_name="sine",
        num_samples=511,
        x_min=-2.0 * math.pi,
        x_max=2.0 * math.pi,
    )

    narrow_spacing = (
        x_narrow[1]
        - x_narrow[0]
    )

    wide_spacing = (
        x_wide[1]
        - x_wide[0]
    )

    assert torch.allclose(
        narrow_spacing,
        wide_spacing,
        atol=1e-7,
    )


def test_integer_fourier_features_are_two_pi_periodic() -> None:
    encoding = FourierFeatureEncoding(
        frequencies=[
            1.0,
            2.0,
            10.0,
        ],
        include_input=False,
    )

    x = torch.linspace(
        -math.pi,
        math.pi,
        101,
    ).unsqueeze(1)

    encoded = encoding(x)

    shifted_encoded = encoding(
        x + 2.0 * math.pi
    )

    assert torch.allclose(
        encoded,
        shifted_encoded,
        atol=2e-5,
    )


def test_zone_metrics_use_fixed_absolute_regions() -> None:
    x = torch.tensor(
        [
            [-2.5 * math.pi],
            [-1.5 * math.pi],
            [-0.5 * math.pi],
            [0.5 * math.pi],
            [1.5 * math.pi],
            [2.5 * math.pi],
        ]
    )

    target = torch.zeros_like(x)

    prediction = torch.tensor(
        [
            [3.0],
            [2.0],
            [1.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    metrics = compute_fixed_zone_metrics(
        x=x,
        target=target,
        prediction=prediction,
        zone_boundaries=(
            math.pi,
            2.0 * math.pi,
            3.0 * math.pi,
        ),
    )

    assert metrics["zone_0_mse"] == 1.0
    assert metrics["zone_1_mse"] == 4.0
    assert metrics["zone_2_mse"] == 9.0


def test_symmetry_metrics_recognize_an_odd_function() -> None:
    model = nn.Linear(
        in_features=1,
        out_features=1,
    )

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()

    metrics = compute_symmetry_metrics(
        model=model,
        max_abs_x=3.0,
    )

    assert (
        metrics.odd_residual_relative
        < 1e-10
    )

    assert abs(
        metrics.even_residual_relative
        - 2.0
    ) < 1e-6


def test_affine_fit_recovers_exact_line() -> None:
    x = torch.linspace(
        -3.0,
        3.0,
        100,
    )

    y = 3.0 * x + 2.0

    metrics = fit_affine_function(
        x=x,
        y=y,
    )

    assert abs(
        metrics.slope
        - 3.0
    ) < 1e-6

    assert abs(
        metrics.intercept
        - 2.0
    ) < 1e-6

    assert abs(
        metrics.r_squared
        - 1.0
    ) < 1e-10
