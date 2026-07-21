"""Compare parameter-matched shallow and deep MLPs on Spiral data."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from toy_model.data import sample_spiral
from toy_model.evaluation import evaluate_classification
from toy_model.models import MLP
from toy_model.training import (
    ClassificationTrainConfig,
    fit_classification,
)

from spiral_experiment_utils import (
    create_run_directory,
    plot_decision_boundary,
    plot_logit_margin,
    save_json,
    save_rows_csv,
    serialize_namespace,
    set_seed,
)


@dataclass(frozen=True)
class ModelCondition:
    """One architecture in the controlled depth comparison."""

    name: str
    hidden_dims: tuple[int, ...] | None
    description: str


def build_model_conditions() -> list[ModelCondition]:
    """Return the linear baseline and three matched-budget MLPs."""
    return [
        ModelCondition(
            name="linear",
            hidden_dims=None,
            description=(
                "Linear logits; negative control for nonlinear geometry."
            ),
        ),
        ModelCondition(
            name="shallow",
            hidden_dims=(256,),
            description=(
                "One hidden layer; approximately 1.5k parameters."
            ),
        ),
        ModelCondition(
            name="medium",
            hidden_dims=(36, 36),
            description=(
                "Two hidden layers; approximately 1.5k parameters."
            ),
        ),
        ModelCondition(
            name="deep",
            hidden_dims=(26, 26, 26),
            description=(
                "Three hidden layers; approximately 1.5k parameters."
            ),
        ),
    ]


def build_model(
    condition: ModelCondition,
    *,
    num_classes: int,
) -> nn.Module:
    """Build one classifier without changing the shared training code."""
    if condition.hidden_dims is None:
        return nn.Linear(
            2,
            num_classes,
        )
    return MLP(
        input_dim=2,
        hidden_dims=condition.hidden_dims,
        output_dim=num_classes,
        activation="relu",
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLP depth while keeping parameter counts similar."
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
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--grid-limit",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=260,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/spiral_depth_comparison"),
    )
    return parser.parse_args()


def aggregate_metrics(
    rows: list[dict[str, object]],
    model_conditions: list[ModelCondition],
) -> list[dict[str, object]]:
    """Aggregate mean and sample standard deviation across seeds."""
    numeric_keys = [
        "train_cross_entropy",
        "train_accuracy",
        "train_mean_true_class_margin",
        "test_cross_entropy",
        "test_accuracy",
        "test_mean_true_class_margin",
    ]
    summaries: list[dict[str, object]] = []

    for condition in model_conditions:
        model_rows = [
            row
            for row in rows
            if row["model_name"] == condition.name
        ]
        summary: dict[str, object] = {
            "model_name": condition.name,
            "parameter_count": model_rows[0][
                "parameter_count"
            ],
            "num_seeds": len(model_rows),
        }
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in model_rows
            ]
            summary[f"{key}_mean"] = (
                statistics.fmean(values)
            )
            summary[f"{key}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )
        summaries.append(summary)

    return summaries


def plot_summary(
    summaries: list[dict[str, object]],
    *,
    output_directory: Path,
) -> None:
    """Plot seed-aggregated accuracy and cross-entropy."""
    labels = [
        str(row["model_name"])
        for row in summaries
    ]
    x_positions = np.arange(
        len(labels)
    )

    accuracy_means = [
        float(row["test_accuracy_mean"])
        for row in summaries
    ]
    accuracy_stds = [
        float(row["test_accuracy_std"])
        for row in summaries
    ]

    accuracy_figure, accuracy_axis = plt.subplots(
        figsize=(8.4, 5.2)
    )
    accuracy_axis.bar(
        x_positions,
        accuracy_means,
        yerr=accuracy_stds,
        capsize=5,
    )
    accuracy_axis.set_xticks(
        x_positions,
        labels,
    )
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.set_ylabel("Test accuracy")
    accuracy_axis.set_title(
        "Parameter-matched depth comparison"
    )
    accuracy_axis.grid(
        axis="y",
        alpha=0.25,
    )
    accuracy_figure.tight_layout()
    accuracy_figure.savefig(
        output_directory
        / "test_accuracy_summary.png",
        dpi=175,
        bbox_inches="tight",
    )
    plt.close(accuracy_figure)

    loss_means = [
        float(row["test_cross_entropy_mean"])
        for row in summaries
    ]
    loss_stds = [
        float(row["test_cross_entropy_std"])
        for row in summaries
    ]

    loss_figure, loss_axis = plt.subplots(
        figsize=(8.4, 5.2)
    )
    loss_axis.bar(
        x_positions,
        loss_means,
        yerr=loss_stds,
        capsize=5,
    )
    loss_axis.set_xticks(
        x_positions,
        labels,
    )
    loss_axis.set_ylabel("Test cross-entropy")
    loss_axis.set_title(
        "Parameter-matched depth comparison"
    )
    loss_axis.grid(
        axis="y",
        alpha=0.25,
    )
    loss_figure.tight_layout()
    loss_figure.savefig(
        output_directory
        / "test_cross_entropy_summary.png",
        dpi=175,
        bbox_inches="tight",
    )
    plt.close(loss_figure)


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError(
            "At least one random seed is required."
        )

    run_directory = create_run_directory(
        output_root=args.output_root,
        experiment_name="matched_parameter_depth",
    )
    predictions_directory = (
        run_directory / "predictions"
    )
    predictions_directory.mkdir()

    model_conditions = build_model_conditions()
    metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    common_data_arguments = {
        "num_classes": args.num_classes,
        "turns": args.turns,
        "radius_min": args.radius_min,
        "radius_max": args.radius_max,
        "angular_noise_std": args.angular_noise_std,
    }

    for seed in args.seeds:
        x_train, y_train = sample_spiral(
            samples_per_class=(
                args.train_samples_per_class
            ),
            seed=seed,
            **common_data_arguments,
        )
        x_test, y_test = sample_spiral(
            samples_per_class=(
                args.test_samples_per_class
            ),
            seed=seed + 10_000,
            **common_data_arguments,
        )

        for condition in model_conditions:
            set_seed(seed)
            model = build_model(
                condition,
                num_classes=args.num_classes,
            )
            parameter_count = count_parameters(
                model
            )

            print(
                "\n"
                f"model={condition.name} | "
                f"seed={seed} | "
                f"parameters={parameter_count}"
            )

            train_result = fit_classification(
                model=model,
                x_train=x_train,
                y_train=y_train,
                config=ClassificationTrainConfig(
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=args.device,
                    log_every=0,
                    checkpoint_steps=(
                        0,
                        args.steps,
                    ),
                ),
                verbose=False,
            )
            train_metrics, _, _ = (
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

            row: dict[str, object] = {
                "model_name": condition.name,
                "seed": seed,
                "parameter_count": parameter_count,
                "final_recorded_training_loss": (
                    train_result.final_loss
                ),
                "final_recorded_training_accuracy": (
                    train_result.final_accuracy
                ),
                **train_metrics.to_dict(
                    prefix="train_"
                ),
                **test_metrics.to_dict(
                    prefix="test_"
                ),
            }
            metric_rows.append(row)

            for step, (loss, accuracy) in enumerate(
                zip(
                    train_result.losses,
                    train_result.accuracies,
                ),
                start=1,
            ):
                history_rows.append(
                    {
                        "model_name": condition.name,
                        "seed": seed,
                        "step": step,
                        "train_loss": loss,
                        "train_accuracy": accuracy,
                    }
                )

            model_directory = (
                predictions_directory
                / f"{condition.name}_seed-{seed}"
            )
            model_directory.mkdir()

            torch.save(
                {
                    "model_name": condition.name,
                    "hidden_dims": (
                        list(condition.hidden_dims)
                        if condition.hidden_dims
                        is not None
                        else None
                    ),
                    "num_classes": args.num_classes,
                    "model_state_dict": (
                        train_result.snapshots[
                            args.steps
                        ]
                    ),
                },
                model_directory / "model.pt",
            )
            save_json(
                {
                    "metrics": row,
                    "test_confusion_matrix": (
                        test_confusion.tolist()
                    ),
                },
                model_directory / "metrics.json",
            )
            plot_decision_boundary(
                model=model,
                x=x_test,
                y=y_test,
                title=(
                    f"{condition.name} | "
                    f"seed={seed} | "
                    f"test accuracy="
                    f"{test_metrics.accuracy:.3f}"
                ),
                output_path=(
                    model_directory
                    / "decision_boundary.png"
                ),
                grid_limit=args.grid_limit,
                grid_points=args.grid_points,
            )
            plot_logit_margin(
                model=model,
                x=x_test,
                y=y_test,
                title=(
                    f"{condition.name} | "
                    f"seed={seed} | "
                    "top-two logit margin"
                ),
                output_path=(
                    model_directory
                    / "logit_margin.png"
                ),
                grid_limit=args.grid_limit,
                grid_points=args.grid_points,
            )

            print(
                f"train accuracy="
                f"{train_metrics.accuracy:.4f} | "
                f"test accuracy="
                f"{test_metrics.accuracy:.4f} | "
                f"test true margin="
                f"{test_metrics.mean_true_class_margin:.4f}"
            )

    summaries = aggregate_metrics(
        rows=metric_rows,
        model_conditions=model_conditions,
    )
    save_rows_csv(
        rows=metric_rows,
        output_path=run_directory / "metrics.csv",
    )
    save_rows_csv(
        rows=history_rows,
        output_path=(
            run_directory
            / "training_history.csv"
        ),
    )
    save_rows_csv(
        rows=summaries,
        output_path=(
            run_directory
            / "summary.csv"
        ),
    )
    save_json(
        {
            "research_question": (
                "At approximately matched parameter counts, "
                "does allocating capacity to greater MLP depth "
                "make a Spiral decision boundary easier to learn?"
            ),
            "hypothesis": (
                "The linear model should fail structurally. "
                "Deeper MLPs may represent the curved boundary "
                "more efficiently, but can also be more sensitive "
                "to optimization."
            ),
            "changed_variable": (
                "Depth and width allocation under an approximately "
                "1.5k-parameter budget."
            ),
            "controlled_variables": [
                "Spiral data generator",
                "training and test sample counts",
                "ReLU activation for all nonlinear models",
                "Adam optimizer",
                "learning rate",
                "training steps",
                "full-batch optimization",
                "evaluation grid",
            ],
            "arguments": serialize_namespace(args),
            "model_conditions": [
                asdict(condition)
                for condition in model_conditions
            ],
            "summaries": summaries,
        },
        run_directory / "experiment_design_and_results.json",
    )
    plot_summary(
        summaries=summaries,
        output_directory=(
            run_directory / "figures"
        ),
    )

    print("\nAggregated results")
    for summary in summaries:
        print(
            f"{summary['model_name']:>8s} | "
            f"parameters="
            f"{int(summary['parameter_count']):4d} | "
            f"test accuracy="
            f"{float(summary['test_accuracy_mean']):.4f}"
            f" ± "
            f"{float(summary['test_accuracy_std']):.4f}"
        )
    print(
        f"Results saved to: {run_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
