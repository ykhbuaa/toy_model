"""Train an MLP to fit a synthetic one-dimensional function."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib

# Use a non-interactive backend on remote servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from toy_model.data import evaluate_function, sample_function
from toy_model.models import MLP
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
    predict,
)


FUNCTION_CHOICES = [
    "linear",
    "sine",
    "high_frequency_sine",
    "mixed_sine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a one-dimensional function with an MLP."
    )

    parser.add_argument(
        "--function",
        type=str,
        default="mixed_sine",
        choices=FUNCTION_CHOICES,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
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
        "--plot-min",
        type=float,
        default=-2.0 * math.pi,
    )
    parser.add_argument(
        "--plot-max",
        type=float,
        default=2.0 * math.pi,
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=2000,
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
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=[
            0,
            10,
            100,
            500,
            1000,
            5000,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/function_fitting"),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set all random seeds used by this experiment."""
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
    """Create a unique directory for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    run_directory = (
        output_root
        / f"{timestamp}_{function_name}_seed-{seed}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    (run_directory / "figures").mkdir()
    (run_directory / "checkpoints").mkdir()

    return run_directory


def serialize_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Convert command-line arguments into JSON-compatible values."""
    result: dict[str, object] = {}

    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        else:
            result[name] = value

    return result


def save_loss_history(
    losses: list[float],
    output_path: Path,
) -> None:
    """Save per-step training losses as CSV."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_loss"])

        for step, loss in enumerate(losses, start=1):
            writer.writerow([step, loss])


def plot_loss_curve(
    losses: list[float],
    output_path: Path,
) -> None:
    """Plot the training loss curve."""
    figure, axis = plt.subplots(figsize=(9, 4.5))

    steps = np.arange(
        1,
        len(losses) + 1,
    )

    axis.plot(
        steps,
        losses,
    )
    axis.set_yscale("log")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Training loss")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_checkpoint_predictions(
    model: MLP,
    snapshots: dict[int, dict[str, torch.Tensor]],
    x_grid: torch.Tensor,
    y_true: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_min: float,
    train_max: float,
    function_name: str,
    output_path: Path,
) -> None:
    """Visualize how the learned function changes during training."""
    checkpoint_steps = sorted(snapshots)

    columns = 2
    rows = math.ceil(len(checkpoint_steps) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 3.7 * rows),
        sharex=True,
    )

    axes_array = np.asarray(axes).reshape(-1)

    model = model.cpu()

    for axis, step in zip(
        axes_array,
        checkpoint_steps,
    ):
        model.load_state_dict(snapshots[step])
        prediction = predict(model, x_grid)

        axis.plot(
            x_grid.squeeze(1).numpy(),
            y_true.squeeze(1).numpy(),
            label="Ground truth",
        )
        axis.plot(
            x_grid.squeeze(1).numpy(),
            prediction.squeeze(1).numpy(),
            label="MLP prediction",
        )
        axis.scatter(
            x_train.squeeze(1).numpy(),
            y_train.squeeze(1).numpy(),
            s=9,
            alpha=0.35,
            label="Training samples",
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
        )

        axis.set_title(f"Step {step}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.2)

    for unused_axis in axes_array[len(checkpoint_steps):]:
        unused_axis.axis("off")

    axes_array[0].legend()

    figure.suptitle(
        f"Learning process: {function_name}",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def compute_region_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    x_grid: torch.Tensor,
    train_min: float,
    train_max: float,
) -> dict[str, float]:
    """Compute interpolation and extrapolation errors."""
    x_values = x_grid.squeeze(1)

    interpolation_mask = (
        (x_values >= train_min)
        & (x_values <= train_max)
    )
    extrapolation_mask = ~interpolation_mask

    squared_error = (
        prediction.squeeze(1)
        - target.squeeze(1)
    ) ** 2

    interpolation_mse = float(
        squared_error[interpolation_mask].mean().item()
    )

    if extrapolation_mask.any():
        extrapolation_mse = float(
            squared_error[extrapolation_mask].mean().item()
        )
    else:
        extrapolation_mse = float("nan")

    return {
        "interpolation_mse": interpolation_mse,
        "extrapolation_mse": extrapolation_mse,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_directory = create_run_directory(
        output_root=args.output_root,
        function_name=args.function,
        seed=args.seed,
    )

    print(f"Run directory: {run_directory.resolve()}")

    x_train, y_train = sample_function(
        function_name=args.function,
        num_samples=args.num_samples,
        x_min=args.train_min,
        x_max=args.train_max,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    model = MLP(
        input_dim=1,
        hidden_dims=args.hidden_dims,
        output_dim=1,
        activation=args.activation,
    )

    print(model)
    print(f"Trainable parameters: {model.count_parameters():,}")

    training_config = RegressionTrainConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        log_every=args.log_every,
        checkpoint_steps=tuple(args.checkpoint_steps),
    )

    training_result = fit_regression(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=training_config,
    )

    x_grid = torch.linspace(
        args.plot_min,
        args.plot_max,
        args.grid_points,
    ).unsqueeze(1)

    y_true = evaluate_function(
        x=x_grid,
        function_name=args.function,
    )

    final_prediction = predict(
        model=model,
        x=x_grid,
    )

    region_metrics = compute_region_metrics(
        prediction=final_prediction,
        target=y_true,
        x_grid=x_grid,
        train_min=args.train_min,
        train_max=args.train_max,
    )

    metrics = {
        "final_train_loss": training_result.final_loss,
        "interpolation_mse": region_metrics["interpolation_mse"],
        "extrapolation_mse": region_metrics["extrapolation_mse"],
        "parameter_count": model.count_parameters(),
        "device": training_result.device,
    }

    config_path = run_directory / "config.json"
    metrics_path = run_directory / "metrics.json"
    loss_history_path = run_directory / "loss_history.csv"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialize_arguments(args),
            file,
            indent=2,
        )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    save_loss_history(
        losses=training_result.losses,
        output_path=loss_history_path,
    )

    for step, state_dict in training_result.snapshots.items():
        checkpoint_path = (
            run_directory
            / "checkpoints"
            / f"step_{step:06d}.pt"
        )

        torch.save(
            {
                "step": step,
                "function": args.function,
                "model_state_dict": state_dict,
                "hidden_dims": args.hidden_dims,
                "activation": args.activation,
            },
            checkpoint_path,
        )

    plot_loss_curve(
        losses=training_result.losses,
        output_path=(
            run_directory
            / "figures"
            / "loss_curve.png"
        ),
    )

    plot_checkpoint_predictions(
        model=model,
        snapshots=training_result.snapshots,
        x_grid=x_grid,
        y_true=y_true,
        x_train=x_train,
        y_train=y_train,
        train_min=args.train_min,
        train_max=args.train_max,
        function_name=args.function,
        output_path=(
            run_directory
            / "figures"
            / "prediction_checkpoints.png"
        ),
    )

    print("\nFinal results")
    print(f"Device: {metrics['device']}")
    print(f"Final train loss: {metrics['final_train_loss']:.8f}")
    print(f"Interpolation MSE: {metrics['interpolation_mse']:.8f}")
    print(f"Extrapolation MSE: {metrics['extrapolation_mse']:.8f}")
    print(f"Results saved to: {run_directory.resolve()}")


if __name__ == "__main__":
    main()
