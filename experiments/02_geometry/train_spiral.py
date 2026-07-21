"""Train one MLP classifier on the synthetic Spiral dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from toy_model.data import sample_spiral
from toy_model.evaluation import evaluate_classification
from toy_model.models import MLP
from toy_model.training import (
    ClassificationTrainConfig,
    fit_classification,
)

from spiral_experiment_utils import (
    create_run_directory,
    plot_checkpoint_boundaries,
    plot_confusion_matrix,
    plot_dataset,
    plot_decision_boundary,
    plot_logit_margin,
    plot_training_curves,
    save_json,
    save_training_history,
    serialize_namespace,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one MLP on a three-class Spiral dataset."
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
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[36, 36],
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "silu", "gelu"],
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
        default=[0, 10, 100, 500, 1000, 5000],
    )
    parser.add_argument(
        "--grid-limit",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/spiral_single"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_directory = create_run_directory(
        output_root=args.output_root,
        experiment_name=f"seed-{args.seed}",
    )

    common_data_arguments = {
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
        **common_data_arguments,
    )
    x_test, y_test = sample_spiral(
        samples_per_class=(
            args.test_samples_per_class
        ),
        seed=args.seed + 10_000,
        **common_data_arguments,
    )

    model = MLP(
        input_dim=2,
        hidden_dims=args.hidden_dims,
        output_dim=args.num_classes,
        activation=args.activation,
    )
    print(model)
    print(
        f"Trainable parameters: "
        f"{model.count_parameters():,}"
    )

    train_config = ClassificationTrainConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        log_every=args.log_every,
        checkpoint_steps=tuple(
            args.checkpoint_steps
        ),
    )
    train_result = fit_classification(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=train_config,
    )

    train_metrics, _, train_confusion = (
        evaluate_classification(
            model=model,
            x=x_train,
            targets=y_train,
        )
    )
    test_metrics, _, test_confusion = (
        evaluate_classification(
            model=model,
            x=x_test,
            targets=y_test,
        )
    )

    metrics = {
        "parameter_count": model.count_parameters(),
        "device": train_result.device,
        **train_metrics.to_dict(prefix="train_"),
        **test_metrics.to_dict(prefix="test_"),
        "train_confusion_matrix": (
            train_confusion.tolist()
        ),
        "test_confusion_matrix": (
            test_confusion.tolist()
        ),
    }

    save_json(
        serialize_namespace(args),
        run_directory / "config.json",
    )
    save_json(
        metrics,
        run_directory / "metrics.json",
    )
    save_training_history(
        losses=train_result.losses,
        accuracies=train_result.accuracies,
        output_path=(
            run_directory
            / "training_history.csv"
        ),
    )

    for step, state_dict in (
        train_result.snapshots.items()
    ):
        torch.save(
            {
                "step": step,
                "model_state_dict": state_dict,
                "hidden_dims": args.hidden_dims,
                "activation": args.activation,
                "num_classes": args.num_classes,
            },
            (
                run_directory
                / "checkpoints"
                / f"step_{step:06d}.pt"
            ),
        )

    torch.save(
        {
            "model_state_dict": (
                train_result.snapshots[args.steps]
            ),
            "hidden_dims": args.hidden_dims,
            "activation": args.activation,
            "num_classes": args.num_classes,
        },
        run_directory / "model.pt",
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
    plot_training_curves(
        losses=train_result.losses,
        accuracies=train_result.accuracies,
        loss_output_path=(
            run_directory
            / "figures"
            / "training_loss.png"
        ),
        accuracy_output_path=(
            run_directory
            / "figures"
            / "training_accuracy.png"
        ),
    )
    plot_decision_boundary(
        model=model,
        x=x_test,
        y=y_test,
        title=(
            "Spiral decision boundary "
            f"(test accuracy={test_metrics.accuracy:.3f})"
        ),
        output_path=(
            run_directory
            / "figures"
            / "decision_boundary.png"
        ),
        grid_limit=args.grid_limit,
        grid_points=args.grid_points,
    )
    plot_logit_margin(
        model=model,
        x=x_test,
        y=y_test,
        title="Top-two logit margin",
        output_path=(
            run_directory
            / "figures"
            / "logit_margin.png"
        ),
        grid_limit=args.grid_limit,
        grid_points=args.grid_points,
    )
    plot_confusion_matrix(
        confusion_matrix=test_confusion,
        title="Test confusion matrix",
        output_path=(
            run_directory
            / "figures"
            / "test_confusion_matrix.png"
        ),
    )
    plot_checkpoint_boundaries(
        model=model,
        snapshots=train_result.snapshots,
        x=x_train,
        y=y_train,
        output_path=(
            run_directory
            / "figures"
            / "decision_boundary_checkpoints.png"
        ),
        grid_limit=args.grid_limit,
        grid_points=min(
            args.grid_points,
            220,
        ),
    )

    print("\nFinal results")
    print(
        f"Train accuracy: {train_metrics.accuracy:.4f}"
    )
    print(
        f"Test accuracy: {test_metrics.accuracy:.4f}"
    )
    print(
        "Test true-class margin: "
        f"{test_metrics.mean_true_class_margin:.4f}"
    )
    print(
        f"Results saved to: {run_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
