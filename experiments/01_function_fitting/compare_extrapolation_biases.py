"""Compare activation and Fourier-feature extrapolation biases.

This experiment separates three questions:

1. Fixed sample budget:
   Does wider coverage help despite lower density?

2. Fixed density:
   What changes when wider coverage receives more samples?

3. Model bias:
   How do ReLU, Tanh and periodic Fourier features extrapolate?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from toy_model.data import (
    evaluate_function,
    sample_function_grid,
)
from toy_model.evaluation import (
    compute_boundary_distance_metrics,
    compute_fixed_zone_metrics,
    compute_periodicity_residual,
    compute_support_metrics,
    compute_symmetry_metrics,
    compute_tail_affine_metrics,
)
from toy_model.models import (
    FourierFeatureMLP,
    MLP,
)
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
    predict,
)


@dataclass(frozen=True)
class DataCondition:
    """One controlled training-data condition."""

    name: str
    train_radius: float
    num_samples: int
    description: str

    @property
    def sample_spacing(self) -> float:
        """Return the spacing of the inclusive linspace grid."""
        return (
            2.0 * self.train_radius
            / (self.num_samples - 1)
        )


@dataclass(frozen=True)
class ModelCondition:
    """Description of one model family."""

    name: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ReLU, Tanh and Fourier-feature "
            "MLP extrapolation."
        )
    )

    parser.add_argument(
        "--function",
        type=str,
        default="mixed_sine",
        choices=[
            "sine",
            "high_frequency_sine",
            "mixed_sine",
        ],
    )

    parser.add_argument(
        "--base-samples",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[
            128,
            128,
            128,
        ],
    )

    parser.add_argument(
        "--fourier-frequencies",
        type=float,
        nargs="+",
        default=[
            float(index)
            for index in range(1, 11)
        ],
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=6145,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/extrapolation_bias"
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_data_conditions(
    base_samples: int,
) -> list[DataCondition]:
    """Create the three controlled data conditions."""
    if base_samples < 2:
        raise ValueError(
            "base_samples must be at least two."
        )

    narrow_radius = math.pi
    wide_radius = 2.0 * math.pi

    base_spacing = (
        2.0 * narrow_radius
        / (base_samples - 1)
    )

    wide_fixed_density_samples = int(
        round(
            2.0 * wide_radius
            / base_spacing
        )
    ) + 1

    return [
        DataCondition(
            name="narrow_base",
            train_radius=narrow_radius,
            num_samples=base_samples,
            description=(
                "Baseline: [-pi, pi] with the "
                "base sample count."
            ),
        ),
        DataCondition(
            name="wide_fixed_samples",
            train_radius=wide_radius,
            num_samples=base_samples,
            description=(
                "Wider support with the same sample "
                "budget; density is reduced."
            ),
        ),
        DataCondition(
            name="wide_fixed_density",
            train_radius=wide_radius,
            num_samples=wide_fixed_density_samples,
            description=(
                "Wider support with the baseline "
                "sample spacing preserved."
            ),
        ),
    ]


def build_model_conditions() -> list[ModelCondition]:
    """Create the three model families."""
    return [
        ModelCondition(
            name="relu_mlp",
            description=(
                "Raw coordinate MLP with ReLU activations."
            ),
        ),
        ModelCondition(
            name="tanh_mlp",
            description=(
                "Raw coordinate MLP with Tanh activations."
            ),
        ),
        ModelCondition(
            name="fourier_mlp",
            description=(
                "MLP on integer sinusoidal features; "
                "exactly 2*pi-periodic."
            ),
        ),
    ]


def build_model(
    model_name: str,
    hidden_dims: list[int],
    fourier_frequencies: list[float],
) -> nn.Module:
    """Build one model for the controlled comparison."""
    if model_name == "relu_mlp":
        return MLP(
            input_dim=1,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
        )

    if model_name == "tanh_mlp":
        return MLP(
            input_dim=1,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="tanh",
        )

    if model_name == "fourier_mlp":
        return FourierFeatureMLP(
            frequencies=fourier_frequencies,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            include_input=False,
        )

    raise ValueError(
        f"Unknown model: {model_name!r}"
    )


def create_run_directory(
    output_root: Path,
    function_name: str,
) -> Path:
    """Create a timestamped result directory."""
    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    run_directory = (
        output_root
        / f"{function_name}_{timestamp}"
    )

    (
        run_directory
        / "predictions"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def plot_prediction(
    *,
    x_grid: torch.Tensor,
    y_true: torch.Tensor,
    prediction: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_radius: float,
    model_name: str,
    condition_name: str,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Plot one trained function over the common evaluation interval."""
    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    axis.plot(
        x_grid.squeeze(1).numpy(),
        y_true.squeeze(1).numpy(),
        label="Ground truth",
        linewidth=2.0,
    )

    axis.plot(
        x_grid.squeeze(1).numpy(),
        prediction.squeeze(1).numpy(),
        label="Prediction",
        linewidth=1.7,
    )

    axis.scatter(
        x_train.squeeze(1).numpy(),
        y_train.squeeze(1).numpy(),
        label="Training samples",
        s=12,
        alpha=0.55,
    )

    axis.axvspan(
        -train_radius,
        train_radius,
        alpha=0.08,
        label="Training support",
    )

    for boundary in [
        -2.0 * math.pi,
        -math.pi,
        math.pi,
        2.0 * math.pi,
    ]:
        axis.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")

    axis.set_title(
        f"{model_name} | {condition_name}\n"
        f"Zone MSE: "
        f"{metrics['zone_0_mse']:.2e}, "
        f"{metrics['zone_1_mse']:.2e}, "
        f"{metrics['zone_2_mse']:.2e}"
    )

    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_zone_summary(
    rows: list[
        dict[str, float | int | str]
    ],
    output_path: Path,
) -> None:
    """Plot Zone 0, Zone 1 and Zone 2 errors."""
    labels = [
        (
            f"{row['model_name']}\n"
            f"{row['data_condition']}\n"
            f"seed={row['seed']}"
        )
        for row in rows
    ]

    x_positions = np.arange(
        len(rows)
    )

    width = 0.26

    figure, axis = plt.subplots(
        figsize=(
            max(13, len(rows) * 1.2),
            6,
        )
    )

    axis.bar(
        x_positions - width,
        [
            float(row["zone_0_mse"])
            for row in rows
        ],
        width=width,
        label="Zone 0: |x| <= pi",
    )

    axis.bar(
        x_positions,
        [
            float(row["zone_1_mse"])
            for row in rows
        ],
        width=width,
        label="Zone 1: pi < |x| <= 2pi",
    )

    axis.bar(
        x_positions + width,
        [
            float(row["zone_2_mse"])
            for row in rows
        ],
        width=width,
        label="Zone 2: 2pi < |x| <= 3pi",
    )

    axis.set_yscale("log")
    axis.set_ylabel("Mean squared error")
    axis.set_title(
        "Error on fixed absolute evaluation zones"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_boundary_summary(
    rows: list[
        dict[str, float | int | str]
    ],
    output_path: Path,
) -> None:
    """Plot error as a function of distance beyond support."""
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    series_keys = sorted(
        {
            (
                str(row["model_name"]),
                str(row["data_condition"]),
                int(row["seed"]),
            )
            for row in rows
        }
    )

    for (
        model_name,
        data_condition,
        seed,
    ) in series_keys:
        series = [
            row
            for row in rows
            if row["model_name"] == model_name
            and row["data_condition"] == data_condition
            and row["seed"] == seed
        ]

        series.sort(
            key=lambda row: float(
                row["distance_min_pi"]
            )
        )

        centers = [
            0.5
            * (
                float(
                    row["distance_min_pi"]
                )
                + float(
                    row["distance_max_pi"]
                )
            )
            for row in series
        ]

        mse_values = [
            float(row["mse"])
            for row in series
        ]

        axis.plot(
            centers,
            mse_values,
            marker="o",
            label=(
                f"{model_name} | "
                f"{data_condition} | "
                f"seed={seed}"
            ),
        )

    axis.set_yscale("log")

    axis.set_xlabel(
        "Distance beyond training boundary "
        "(multiples of pi)"
    )

    axis.set_ylabel(
        "Mean squared error"
    )

    axis.set_title(
        "Extrapolation error at equal distance from support"
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_csv(
    rows: list[
        dict[str, float | int | str]
    ],
    output_path: Path,
) -> None:
    """Save a list of dictionaries as CSV."""
    if not rows:
        raise ValueError(
            "Cannot save an empty CSV file."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise ValueError(
            "steps must be positive."
        )

    if args.grid_points < 3:
        raise ValueError(
            "grid_points must be at least three."
        )

    if not args.seeds:
        raise ValueError(
            "At least one seed is required."
        )

    data_conditions = build_data_conditions(
        args.base_samples
    )

    model_conditions = (
        build_model_conditions()
    )

    run_directory = create_run_directory(
        output_root=args.output_root,
        function_name=args.function,
    )

    evaluation_radius = (
        3.0 * math.pi
    )

    x_grid = torch.linspace(
        -evaluation_radius,
        evaluation_radius,
        args.grid_points,
    ).unsqueeze(1)

    y_true = evaluate_function(
        x=x_grid,
        function_name=args.function,
    )

    metric_rows: list[
        dict[str, float | int | str]
    ] = []

    boundary_rows: list[
        dict[str, float | int | str]
    ] = []

    for seed in args.seeds:
        for model_condition in model_conditions:
            for data_condition in data_conditions:
                print(
                    "\n"
                    f"model={model_condition.name} | "
                    f"condition={data_condition.name} | "
                    f"seed={seed} | "
                    f"samples={data_condition.num_samples} | "
                    f"spacing="
                    f"{data_condition.sample_spacing:.6f}"
                )

                x_train, y_train = (
                    sample_function_grid(
                        function_name=args.function,
                        num_samples=(
                            data_condition.num_samples
                        ),
                        x_min=(
                            -data_condition.train_radius
                        ),
                        x_max=(
                            data_condition.train_radius
                        ),
                    )
                )

                # For one architecture and one seed, all data conditions
                # start from exactly the same parameter initialization.
                set_seed(seed)

                model = build_model(
                    model_name=model_condition.name,
                    hidden_dims=args.hidden_dims,
                    fourier_frequencies=(
                        args.fourier_frequencies
                    ),
                )

                train_config = RegressionTrainConfig(
                    steps=args.steps,
                    learning_rate=(
                        args.learning_rate
                    ),
                    weight_decay=args.weight_decay,
                    device=args.device,
                    log_every=0,
                    checkpoint_steps=(
                        0,
                        args.steps,
                    ),
                )

                train_result = fit_regression(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    config=train_config,
                    verbose=False,
                )

                prediction = predict(
                    model=model,
                    x=x_grid,
                )

                zone_metrics = (
                    compute_fixed_zone_metrics(
                        x=x_grid,
                        target=y_true,
                        prediction=prediction,
                        zone_boundaries=(
                            math.pi,
                            2.0 * math.pi,
                            3.0 * math.pi,
                        ),
                    )
                )

                support_metrics = (
                    compute_support_metrics(
                        x=x_grid,
                        target=y_true,
                        prediction=prediction,
                        train_radius=(
                            data_condition.train_radius
                        ),
                    )
                )

                symmetry_metrics = (
                    compute_symmetry_metrics(
                        model=model,
                        max_abs_x=evaluation_radius,
                    )
                )

                periodicity_residual = (
                    compute_periodicity_residual(
                        model=model,
                        period=2.0 * math.pi,
                        x_min=-math.pi,
                        x_max=math.pi,
                    )
                )

                (
                    left_tail,
                    right_tail,
                ) = compute_tail_affine_metrics(
                    x=x_grid,
                    prediction=prediction,
                    inner_abs_x=2.0 * math.pi,
                    outer_abs_x=3.0 * math.pi,
                )

                metric_row: dict[
                    str,
                    float | int | str,
                ] = {
                    "model_name": (
                        model_condition.name
                    ),
                    "data_condition": (
                        data_condition.name
                    ),
                    "seed": seed,
                    "train_radius_pi": (
                        data_condition.train_radius
                        / math.pi
                    ),
                    "num_samples": (
                        data_condition.num_samples
                    ),
                    "sample_spacing": (
                        data_condition.sample_spacing
                    ),
                    "parameter_count": (
                        count_parameters(model)
                    ),
                    "final_training_loss": (
                        train_result.final_loss
                    ),
                    **zone_metrics,
                    **support_metrics,
                    **symmetry_metrics.to_dict(),
                    (
                        "periodicity_residual_"
                        "relative"
                    ): periodicity_residual,
                    **left_tail.to_dict(
                        "left_tail"
                    ),
                    **right_tail.to_dict(
                        "right_tail"
                    ),
                }

                metric_rows.append(
                    metric_row
                )

                distance_metrics = (
                    compute_boundary_distance_metrics(
                        x=x_grid,
                        target=y_true,
                        prediction=prediction,
                        train_radius=(
                            data_condition.train_radius
                        ),
                        distance_edges=(
                            0.0,
                            0.25 * math.pi,
                            0.50 * math.pi,
                            0.75 * math.pi,
                            1.00 * math.pi,
                        ),
                    )
                )

                for (
                    distance_metric
                ) in distance_metrics:
                    boundary_rows.append(
                        {
                            "model_name": (
                                model_condition.name
                            ),
                            "data_condition": (
                                data_condition.name
                            ),
                            "seed": seed,
                            "train_radius_pi": (
                                data_condition.train_radius
                                / math.pi
                            ),
                            "distance_min_pi": (
                                distance_metric.distance_min
                                / math.pi
                            ),
                            "distance_max_pi": (
                                distance_metric.distance_max
                                / math.pi
                            ),
                            "mse": (
                                distance_metric.mse
                            ),
                            "num_points": (
                                distance_metric.num_points
                            ),
                        }
                    )

                plot_prediction(
                    x_grid=x_grid,
                    y_true=y_true,
                    prediction=prediction,
                    x_train=x_train,
                    y_train=y_train,
                    train_radius=(
                        data_condition.train_radius
                    ),
                    model_name=(
                        model_condition.name
                    ),
                    condition_name=(
                        data_condition.name
                    ),
                    metrics=zone_metrics,
                    output_path=(
                        run_directory
                        / "predictions"
                        / (
                            f"{model_condition.name}_"
                            f"{data_condition.name}_"
                            f"seed-{seed}.png"
                        )
                    ),
                )

                print(
                    f"final loss="
                    f"{train_result.final_loss:.3e} | "
                    f"zones=("
                    f"{zone_metrics['zone_0_mse']:.3e}, "
                    f"{zone_metrics['zone_1_mse']:.3e}, "
                    f"{zone_metrics['zone_2_mse']:.3e}) | "
                    f"odd residual="
                    f"{symmetry_metrics.odd_residual_relative:.3e} | "
                    f"period residual="
                    f"{periodicity_residual:.3e}"
                )

    save_csv(
        rows=metric_rows,
        output_path=(
            run_directory
            / "metrics.csv"
        ),
    )

    save_csv(
        rows=boundary_rows,
        output_path=(
            run_directory
            / "boundary_distance_metrics.csv"
        ),
    )

    plot_zone_summary(
        rows=metric_rows,
        output_path=(
            run_directory
            / "zone_mse.png"
        ),
    )

    plot_boundary_summary(
        rows=boundary_rows,
        output_path=(
            run_directory
            / "boundary_distance_mse.png"
        ),
    )

    config = {
        "arguments": (
            vars(args)
            | {
                "output_root": str(
                    args.output_root
                )
            }
        ),
        "data_conditions": [
            asdict(item)
            for item in data_conditions
        ],
        "model_conditions": [
            asdict(item)
            for item in model_conditions
        ],
        "fixed_sample_comparison": [
            "narrow_base",
            "wide_fixed_samples",
        ],
        "fixed_density_comparison": [
            "narrow_base",
            "wide_fixed_density",
        ],
    }

    with (
        run_directory
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print(
        "\nResults saved to: "
        f"{run_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
