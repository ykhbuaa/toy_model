"""Analyze the frequency components learned by an MLP over time."""

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

from toy_model.analysis import (
    compute_amplitude_spectrum,
    find_nearest_frequency_amplitude,
)
from toy_model.data import (
    evaluate_function,
    sample_function,
)
from toy_model.models import MLP
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze spectral bias during MLP training."
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
        "--num-samples",
        type=int,
        default=256,
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
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[
            0,
            10,
            100,
            200,
            500,
            800,
            1000,
            2000,
            5000,
        ],
    )
    parser.add_argument(
        "--fft-points",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=20.0,
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
        default=Path("runs/spectral_bias"),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_fft_grid(
    x_min: float,
    x_max: float,
    num_points: int,
) -> torch.Tensor:
    """Create a uniform grid without duplicating the periodic endpoint."""
    if num_points <= 1:
        raise ValueError("num_points must be greater than one.")

    spacing = (
        x_max - x_min
    ) / num_points

    return (
        x_min
        + torch.arange(num_points)
        * spacing
    ).unsqueeze(1)


def create_run_directory(
    output_root: Path,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    run_directory = (
        output_root
        / f"{timestamp}_seed-{seed}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def plot_frequency_spectra(
    checkpoint_data: list[dict[str, object]],
    true_frequencies: torch.Tensor,
    true_amplitudes: torch.Tensor,
    max_frequency: float,
    output_path: Path,
) -> None:
    columns = 2
    rows = math.ceil(
        len(checkpoint_data) / columns
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 3.6 * rows),
    )

    axes_flat = np.asarray(axes).reshape(-1)

    true_amplitudes_plot = torch.clamp(
        true_amplitudes,
        min=1e-7,
    )

    for axis, result in zip(
        axes_flat,
        checkpoint_data,
    ):
        step = int(result["step"])
        frequencies = result["frequencies"]
        amplitudes = result["amplitudes"]

        amplitudes_plot = torch.clamp(
            amplitudes,
            min=1e-7,
        )

        frequency_mask = (
            true_frequencies <= max_frequency
        )

        axis.plot(
            true_frequencies[frequency_mask].numpy(),
            true_amplitudes_plot[frequency_mask].numpy(),
            label="Ground truth",
        )
        axis.plot(
            frequencies[frequency_mask].numpy(),
            amplitudes_plot[frequency_mask].numpy(),
            label="MLP prediction",
        )

        axis.set_yscale("log")
        axis.set_xlim(
            0.0,
            max_frequency,
        )
        axis.set_ylim(
            1e-6,
            2.0,
        )
        axis.set_title(f"Step {step}")
        axis.set_xlabel("Angular frequency")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.25)

    for unused_axis in axes_flat[len(checkpoint_data):]:
        unused_axis.axis("off")

    axes_flat[0].legend()

    figure.suptitle(
        "Frequency spectrum during training",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_mode_amplitudes(
    checkpoint_data: list[dict[str, object]],
    true_low_amplitude: float,
    true_high_amplitude: float,
    output_path: Path,
) -> None:
    steps = [
        int(result["step"])
        for result in checkpoint_data
    ]

    low_amplitudes = [
        float(result["low_amplitude"])
        for result in checkpoint_data
    ]

    high_amplitudes = [
        float(result["high_amplitude"])
        for result in checkpoint_data
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5),
    )

    axis.plot(
        steps,
        low_amplitudes,
        marker="o",
        label="Learned amplitude at ω=1",
    )
    axis.plot(
        steps,
        high_amplitudes,
        marker="o",
        label="Learned amplitude at ω=10",
    )

    axis.axhline(
        true_low_amplitude,
        linestyle="--",
        label="Target amplitude at ω=1",
    )
    axis.axhline(
        true_high_amplitude,
        linestyle=":",
        label="Target amplitude at ω=10",
    )

    axis.set_xscale(
        "symlog",
        linthresh=10,
    )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Amplitude")
    axis.set_title(
        "Learning speed of low- and high-frequency modes"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def save_frequency_metrics(
    checkpoint_data: list[dict[str, object]],
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "step",
                "amplitude_omega_1",
                "amplitude_omega_10",
                "spectrum_mse",
            ]
        )

        for result in checkpoint_data:
            writer.writerow(
                [
                    result["step"],
                    result["low_amplitude"],
                    result["high_amplitude"],
                    result["spectrum_mse"],
                ]
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_directory = create_run_directory(
        output_root=args.output_root,
        seed=args.seed,
    )

    x_train, y_train = sample_function(
        function_name=args.function,
        num_samples=args.num_samples,
        x_min=args.train_min,
        x_max=args.train_max,
        seed=args.seed,
    )

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
        log_every=500,
        checkpoint_steps=tuple(
            args.checkpoint_steps
        ),
    )

    training_result = fit_regression(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=training_config,
    )

    x_fft = create_fft_grid(
        x_min=args.train_min,
        x_max=args.train_max,
        num_points=args.fft_points,
    )

    y_true = evaluate_function(
        x=x_fft,
        function_name=args.function,
    )

    true_frequencies, true_amplitudes = (
        compute_amplitude_spectrum(
            x=x_fft,
            y=y_true,
        )
    )

    true_low_amplitude = (
        find_nearest_frequency_amplitude(
            angular_frequencies=true_frequencies,
            amplitudes=true_amplitudes,
            target_frequency=1.0,
        )
    )

    true_high_amplitude = (
        find_nearest_frequency_amplitude(
            angular_frequencies=true_frequencies,
            amplitudes=true_amplitudes,
            target_frequency=10.0,
        )
    )

    model = model.cpu()

    checkpoint_data: list[dict[str, object]] = []

    for step in sorted(
        training_result.snapshots
    ):
        model.load_state_dict(
            training_result.snapshots[step]
        )

        prediction = predict(
            model=model,
            x=x_fft,
        )

        frequencies, amplitudes = (
            compute_amplitude_spectrum(
                x=x_fft,
                y=prediction,
            )
        )

        low_amplitude = (
            find_nearest_frequency_amplitude(
                angular_frequencies=frequencies,
                amplitudes=amplitudes,
                target_frequency=1.0,
            )
        )

        high_amplitude = (
            find_nearest_frequency_amplitude(
                angular_frequencies=frequencies,
                amplitudes=amplitudes,
                target_frequency=10.0,
            )
        )

        spectrum_mse = float(
            torch.mean(
                (
                    amplitudes
                    - true_amplitudes
                ) ** 2
            ).item()
        )

        checkpoint_data.append(
            {
                "step": step,
                "frequencies": frequencies,
                "amplitudes": amplitudes,
                "low_amplitude": low_amplitude,
                "high_amplitude": high_amplitude,
                "spectrum_mse": spectrum_mse,
            }
        )

        print(
            f"step={step:6d} | "
            f"amp(ω=1)={low_amplitude:.6f} | "
            f"amp(ω=10)={high_amplitude:.6f} | "
            f"spectrum_mse={spectrum_mse:.8f}"
        )

    plot_frequency_spectra(
        checkpoint_data=checkpoint_data,
        true_frequencies=true_frequencies,
        true_amplitudes=true_amplitudes,
        max_frequency=args.max_frequency,
        output_path=(
            run_directory
            / "frequency_spectra.png"
        ),
    )

    plot_mode_amplitudes(
        checkpoint_data=checkpoint_data,
        true_low_amplitude=true_low_amplitude,
        true_high_amplitude=true_high_amplitude,
        output_path=(
            run_directory
            / "mode_amplitudes.png"
        ),
    )

    save_frequency_metrics(
        checkpoint_data=checkpoint_data,
        output_path=(
            run_directory
            / "frequency_metrics.csv"
        ),
    )

    print(f"\nTarget amplitude at ω=1: {true_low_amplitude:.6f}")
    print(f"Target amplitude at ω=10: {true_high_amplitude:.6f}")
    print(f"Results saved to: {run_directory.resolve()}")


if __name__ == "__main__":
    main()
