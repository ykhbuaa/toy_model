"""Compare loss oscillations under different learning rates."""

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
from toy_model.models import MLP
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare regression learning rates."
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
        "--learning-rates",
        type=float,
        nargs="+",
        default=[
            1e-2,
            1e-3,
            3e-4,
            1e-4,
        ],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
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
        "--smoothing-window",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/learning_rate_comparison"
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(
    values: list[float],
    window: int,
) -> np.ndarray:
    values_array = np.asarray(
        values,
        dtype=np.float64,
    )

    if window <= 1:
        return values_array

    if window > len(values_array):
        raise ValueError(
            "Smoothing window cannot exceed loss history length."
        )

    kernel = np.ones(window) / window

    return np.convolve(
        values_array,
        kernel,
        mode="valid",
    )


def calculate_tail_metrics(
    losses: list[float],
) -> dict[str, float]:
    losses_array = np.asarray(
        losses,
        dtype=np.float64,
    )

    tail_start = int(
        0.8 * len(losses_array)
    )
    tail = losses_array[tail_start:]

    safe_tail = np.clip(
        tail,
        1e-16,
        None,
    )

    log_tail = np.log10(safe_tail)

    upward_fraction = float(
        np.mean(
            np.diff(tail) > 0
        )
    )

    return {
        "final_loss": float(losses_array[-1]),
        "minimum_loss": float(losses_array.min()),
        "tail_log_std": float(log_tail.std()),
        "tail_upward_fraction": upward_fraction,
    }


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    run_directory = (
        args.output_root
        / timestamp
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    x_train, y_train = sample_function(
        function_name=args.function,
        num_samples=args.num_samples,
        x_min=-math.pi,
        x_max=math.pi,
        seed=args.seed,
    )

    experiment_results: list[dict[str, object]] = []

    for learning_rate in args.learning_rates:
        # Resetting the seed ensures every model begins from the same
        # initialization. The learning rate is the only changed variable.
        set_seed(args.seed)

        model = MLP(
            input_dim=1,
            hidden_dims=args.hidden_dims,
            output_dim=1,
            activation=args.activation,
        )

        config = RegressionTrainConfig(
            steps=args.steps,
            learning_rate=learning_rate,
            device=args.device,
            log_every=0,
            checkpoint_steps=(0, args.steps),
        )

        training_result = fit_regression(
            model=model,
            x_train=x_train,
            y_train=y_train,
            config=config,
            verbose=False,
        )

        tail_metrics = calculate_tail_metrics(
            training_result.losses
        )

        experiment_results.append(
            {
                "learning_rate": learning_rate,
                "losses": training_result.losses,
                **tail_metrics,
            }
        )

        print(
            f"lr={learning_rate:.1e} | "
            f"final={tail_metrics['final_loss']:.8f} | "
            f"minimum={tail_metrics['minimum_loss']:.8f} | "
            f"tail_log_std={tail_metrics['tail_log_std']:.6f} | "
            f"upward_fraction="
            f"{tail_metrics['tail_upward_fraction']:.3f}"
        )

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    for result in experiment_results:
        losses = result["losses"]
        learning_rate = float(
            result["learning_rate"]
        )

        axis.plot(
            np.arange(1, len(losses) + 1),
            losses,
            label=f"lr={learning_rate:.1e}",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Training step")
    axis.set_ylabel("MSE loss")
    axis.set_title("Raw training losses")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        run_directory / "loss_raw.png",
        dpi=160,
    )
    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    for result in experiment_results:
        losses = result["losses"]
        learning_rate = float(
            result["learning_rate"]
        )

        smoothed_losses = moving_average(
            values=losses,
            window=args.smoothing_window,
        )

        smoothed_steps = np.arange(
            args.smoothing_window,
            len(losses) + 1,
        )

        axis.plot(
            smoothed_steps,
            smoothed_losses,
            label=f"lr={learning_rate:.1e}",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Smoothed MSE loss")
    axis.set_title(
        f"Training losses, moving average window="
        f"{args.smoothing_window}"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        run_directory / "loss_smoothed.png",
        dpi=160,
    )
    plt.close(figure)

    with (
        run_directory
        / "learning_rate_metrics.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "learning_rate",
                "final_loss",
                "minimum_loss",
                "tail_log_std",
                "tail_upward_fraction",
            ]
        )

        for result in experiment_results:
            writer.writerow(
                [
                    result["learning_rate"],
                    result["final_loss"],
                    result["minimum_loss"],
                    result["tail_log_std"],
                    result["tail_upward_fraction"],
                ]
            )

    print(f"Results saved to: {run_directory.resolve()}")


if __name__ == "__main__":
    main()
