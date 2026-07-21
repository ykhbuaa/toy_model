"""Inspect independently sampled Spiral training and test datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from toy_model.data import sample_spiral

from spiral_experiment_utils import (
    create_run_directory,
    plot_dataset,
    save_json,
    serialize_namespace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the synthetic multi-class Spiral dataset."
        )
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--train-samples-per-class",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--test-samples-per-class",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--turns",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--radius-min",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--radius-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--angular-noise-std",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/spiral_inspection"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_directory = create_run_directory(
        output_root=args.output_root,
        experiment_name=f"seed-{args.seed}",
    )

    common_arguments = {
        "num_classes": args.num_classes,
        "turns": args.turns,
        "radius_min": args.radius_min,
        "radius_max": args.radius_max,
        "angular_noise_std": args.angular_noise_std,
    }
    x_train, y_train = sample_spiral(
        samples_per_class=(
            args.train_samples_per_class
        ),
        seed=args.seed,
        **common_arguments,
    )
    x_test, y_test = sample_spiral(
        samples_per_class=(
            args.test_samples_per_class
        ),
        seed=args.seed + 10_000,
        **common_arguments,
    )

    plot_dataset(
        x=x_train,
        y=y_train,
        title="Spiral training dataset",
        output_path=(
            run_directory
            / "figures"
            / "training_dataset.png"
        ),
    )
    plot_dataset(
        x=x_test,
        y=y_test,
        title="Spiral test dataset",
        output_path=(
            run_directory
            / "figures"
            / "test_dataset.png"
        ),
    )
    save_json(
        serialize_namespace(args),
        run_directory / "config.json",
    )

    print(
        f"Training samples: {x_train.shape[0]}"
    )
    print(
        f"Test samples: {x_test.shape[0]}"
    )
    print(
        f"Results saved to: {run_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
