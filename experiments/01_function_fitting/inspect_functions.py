"""Inspect synthetic one-dimensional function datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

# The server may not have a graphical desktop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from toy_model.data import evaluate_function, sample_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a synthetic one-dimensional function dataset."
    )

    parser.add_argument(
        "--function",
        type=str,
        default="mixed_sine",
        choices=[
            "linear",
            "sine",
            "high_frequency_sine",
            "mixed_sine",
        ],
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--x-min", type=float, default=-math.pi)
    parser.add_argument("--x-max", type=float, default=math.pi)
    parser.add_argument("--plot-min", type=float, default=-2.0 * math.pi)
    parser.add_argument("--plot-max", type=float, default=2.0 * math.pi)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-points", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/data_inspection"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_samples, y_samples = sample_function(
        function_name=args.function,
        num_samples=args.num_samples,
        x_min=args.x_min,
        x_max=args.x_max,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    x_grid = torch.linspace(
        args.plot_min,
        args.plot_max,
        args.grid_points,
    ).unsqueeze(1)

    y_grid = evaluate_function(
        x=x_grid,
        function_name=args.function,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = args.output_dir / f"{args.function}.png"

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(
        x_grid.squeeze(1).numpy(),
        y_grid.squeeze(1).numpy(),
        label="Ground-truth function",
    )

    axis.scatter(
        x_samples.squeeze(1).numpy(),
        y_samples.squeeze(1).numpy(),
        s=24,
        label="Training samples",
    )

    axis.axvline(
        args.x_min,
        linestyle="--",
        label="Training boundary",
    )
    axis.axvline(
        args.x_max,
        linestyle="--",
    )

    axis.axvspan(
        args.x_min,
        args.x_max,
        alpha=0.08,
        label="Training interval",
    )

    axis.set_title(
        f"Function: {args.function} | "
        f"samples={args.num_samples} | "
        f"noise_std={args.noise_std}"
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

    print(f"Function: {args.function}")
    print(f"Sample shape: x={tuple(x_samples.shape)}, y={tuple(y_samples.shape)}")
    print(f"Sample x range: [{x_samples.min().item():.4f}, {x_samples.max().item():.4f}]")
    print(f"Saved figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()
