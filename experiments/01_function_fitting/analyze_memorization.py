"""Study interpolation, extrapolation and noise memorization."""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from toy_model.data import sample_function
from toy_model.evaluation import evaluate_function_regression
from toy_model.models import MLP
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
)


FUNCTION_CHOICES = [
    "sine",
    "high_frequency_sine",
    "mixed_sine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare training-point fit, interpolation and extrapolation."
        )
    )

    parser.add_argument(
        "--function",
        type=str,
        default="sine",
        choices=FUNCTION_CHOICES,
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[16, 64, 256],
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.0, 0.2],
    )

    parser.add_argument(
        "--train-min",
        type=float,
        default=-math.pi,
    )
    parser.add_argument(
        "--train-max",
        type=float,
        default=math.pi,
    )
    parser.add_argument(
        "--evaluation-min",
        type=float,
        default=-2.0 * math.pi,
    )
    parser.add_argument(
        "--evaluation-max",
        type=float,
        default=2.0 * math.pi,
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128, 128],
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=[
            "relu",
            "tanh",
            "silu",
            "gelu",
        ],
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=6000,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/memorization"),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory(
    output_root: Path,
    function_name: str,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    run_directory = (
        output_root
        / f"{timestamp}_{function_name}_seed-{seed}_linspace"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    (run_directory / "predictions").mkdir()

    return run_directory


def safe_float_label(value: float) -> str:
    return (
        f"{value:g}"
        .replace("-", "neg")
        .replace(".", "p")
    )


def plot_prediction(
    x_grid: torch.Tensor,
    y_true: torch.Tensor,
    prediction: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_min: float,
    train_max: float,
    sample_count: int,
    noise_std: float,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    axis.plot(
        x_grid.squeeze(1).numpy(),
        y_true.squeeze(1).numpy(),
        label="Clean ground truth",
    )

    axis.plot(
        x_grid.squeeze(1).numpy(),
        prediction.squeeze(1).numpy(),
        label="MLP prediction",
    )

    axis.scatter(
        x_train.squeeze(1).numpy(),
        y_train.squeeze(1).numpy(),
        s=24,
        alpha=0.65,
        label="Observed training labels",
    )

    axis.axvline(
        train_min,
        linestyle="--",
        alpha=0.7,
    )
    axis.axvline(
        train_max,
        linestyle="--",
        alpha=0.7,
    )
    axis.axvspan(
        train_min,
        train_max,
        alpha=0.05,
        label="Training interval",
    )

    axis.set_title(
        f"samples={sample_count}, noise_std={noise_std:g}\n"
        f"train-label MSE={metrics['train_label_mse']:.2e}, "
        f"interpolation MSE={metrics['interpolation_mse']:.2e}, "
        f"extrapolation MSE={metrics['extrapolation_mse']:.2e}"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_metric_summary(
    results: list[dict[str, float]],
    noise_stds: list[float],
    output_path: Path,
) -> None:
    metric_specs = [
        (
            "train_label_mse",
            "Training-label MSE",
        ),
        (
            "interpolation_mse",
            "Interpolation MSE",
        ),
        (
            "extrapolation_mse",
            "Extrapolation MSE",
        ),
        (
            "noise_fit_gap",
            "Noise-fit gap",
        ),
    ]

    figure, axes = plt.subplots(
        len(metric_specs),
        1,
        figsize=(10, 14),
        sharex=True,
    )

    for axis, (metric_name, title) in zip(
        axes,
        metric_specs,
    ):
        for noise_std in noise_stds:
            subset = [
                result
                for result in results
                if result["noise_std"] == noise_std
            ]

            subset.sort(
                key=lambda item: item["sample_count"]
            )

            sample_counts = [
                int(result["sample_count"])
                for result in subset
            ]

            metric_values = [
                max(
                    float(result[metric_name]),
                    1e-12,
                )
                for result in subset
            ]

            axis.plot(
                sample_counts,
                metric_values,
                marker="o",
                label=f"noise_std={noise_std:g}",
            )

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Number of training samples")

    figure.suptitle(
        "Training fit, interpolation and extrapolation",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_results(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    fieldnames = [
        "sample_count",
        "noise_std",
        "final_optimization_loss",
        "train_label_mse",
        "train_clean_mse",
        "interpolation_mse",
        "extrapolation_mse",
        "interpolation_max_abs_error",
        "extrapolation_max_abs_error",
        "noise_fit_gap",
        "interpolation_gap",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()

    if any(
        sample_count <= 0
        for sample_count in args.sample_counts
    ):
        raise ValueError(
            "All sample counts must be positive."
        )

    if any(
        noise_std < 0
        for noise_std in args.noise_stds
    ):
        raise ValueError(
            "Noise standard deviations cannot be negative."
        )

    run_directory = create_run_directory(
        output_root=args.output_root,
        function_name=args.function,
        seed=args.seed,
    )

    results: list[dict[str, float]] = []

    for noise_std in args.noise_stds:
        for sample_count in args.sample_counts:
            print(
                "\n"
                f"Training samples={sample_count}, "
                f"noise_std={noise_std:g}"
            )

            x_train, y_train = sample_function(
                function_name=args.function,
                num_samples=sample_count,
                x_min=args.train_min,
                x_max=args.train_max,
                noise_std=noise_std,
                seed=args.seed,
            )

            # Reset the seed so every comparison starts from the same
            # parameter initialization.
            set_seed(args.seed)

            model = MLP(
                input_dim=1,
                hidden_dims=args.hidden_dims,
                output_dim=1,
                activation=args.activation,
            )

            training_config = RegressionTrainConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
                device=args.device,
                log_every=0,
                checkpoint_steps=(
                    0,
                    args.steps,
                ),
            )

            training_result = fit_regression(
                model=model,
                x_train=x_train,
                y_train=y_train,
                config=training_config,
                verbose=False,
            )

            (
                metric_object,
                x_grid,
                y_true,
                prediction,
            ) = evaluate_function_regression(
                model=model,
                x_train=x_train,
                y_train=y_train,
                function_name=args.function,
                train_min=args.train_min,
                train_max=args.train_max,
                evaluation_min=args.evaluation_min,
                evaluation_max=args.evaluation_max,
                grid_points=args.grid_points,
            )

            metrics = metric_object.to_dict()

            # Positive and large:
            # model is closer to noisy labels than to the clean function
            # at the same training coordinates.
            noise_fit_gap = (
                metrics["train_clean_mse"]
                - metrics["train_label_mse"]
            )

            # Positive and large:
            # model performs worse between training points than exactly
            # at the sampled coordinates.
            interpolation_gap = (
                metrics["interpolation_mse"]
                - metrics["train_clean_mse"]
            )

            result = {
                "sample_count": sample_count,
                "noise_std": noise_std,
                "final_optimization_loss": (
                    training_result.final_loss
                ),
                **metrics,
                "noise_fit_gap": noise_fit_gap,
                "interpolation_gap": interpolation_gap,
            }

            results.append(result)

            noise_label = safe_float_label(
                noise_std
            )

            plot_prediction(
                x_grid=x_grid,
                y_true=y_true,
                prediction=prediction,
                x_train=x_train,
                y_train=y_train,
                train_min=args.train_min,
                train_max=args.train_max,
                sample_count=sample_count,
                noise_std=noise_std,
                metrics=metrics,
                output_path=(
                    run_directory
                    / "predictions"
                    / (
                        f"samples-{sample_count}"
                        f"_noise-{noise_label}.png"
                    )
                ),
            )

            print(
                f"train label MSE: "
                f"{metrics['train_label_mse']:.8f}"
            )
            print(
                f"train clean MSE: "
                f"{metrics['train_clean_mse']:.8f}"
            )
            print(
                f"interpolation MSE: "
                f"{metrics['interpolation_mse']:.8f}"
            )
            print(
                f"extrapolation MSE: "
                f"{metrics['extrapolation_mse']:.8f}"
            )
            print(
                f"noise-fit gap: "
                f"{noise_fit_gap:.8f}"
            )

    plot_metric_summary(
        results=results,
        noise_stds=args.noise_stds,
        output_path=(
            run_directory
            / "metric_summary.png"
        ),
    )

    save_results(
        results=results,
        output_path=(
            run_directory
            / "metrics.csv"
        ),
    )

    print(
        f"\nResults saved to: "
        f"{run_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
