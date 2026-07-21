"""Experiment-local helpers for the Spiral classification studies."""

from __future__ import annotations

import csv
import json
import math
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from toy_model.training import predict_logits


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory(
    output_root: Path,
    experiment_name: str,
) -> Path:
    """Create a unique timestamped experiment directory."""
    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    run_directory = (
        output_root
        / f"{timestamp}_{experiment_name}"
    )
    (run_directory / "figures").mkdir(
        parents=True,
        exist_ok=False,
    )
    (run_directory / "checkpoints").mkdir()
    return run_directory


def serialize_namespace(
    args: Namespace,
) -> dict[str, object]:
    """Convert command-line arguments to JSON-compatible values."""
    result: dict[str, object] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        else:
            result[name] = value
    return result


def save_json(
    value: object,
    output_path: Path,
) -> None:
    """Save a JSON-compatible value."""
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
        )


def save_rows_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save flat dictionaries as a CSV table."""
    if not rows:
        raise ValueError(
            "Cannot save an empty CSV table."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_training_history(
    losses: list[float],
    accuracies: list[float],
    output_path: Path,
    *,
    model_name: str | None = None,
    seed: int | None = None,
) -> None:
    """Save per-step loss and accuracy."""
    rows: list[dict[str, object]] = []
    for step, (loss, accuracy) in enumerate(
        zip(losses, accuracies),
        start=1,
    ):
        row: dict[str, object] = {
            "step": step,
            "train_loss": loss,
            "train_accuracy": accuracy,
        }
        if model_name is not None:
            row["model_name"] = model_name
        if seed is not None:
            row["seed"] = seed
        rows.append(row)
    save_rows_csv(
        rows=rows,
        output_path=output_path,
    )


def plot_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Plot a two-dimensional labeled dataset."""
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    figure, axis = plt.subplots(
        figsize=(7.2, 7.0)
    )
    scatter = axis.scatter(
        x_numpy[:, 0],
        x_numpy[:, 1],
        c=y_numpy,
        s=16,
        alpha=0.8,
    )
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    legend = axis.legend(
        *scatter.legend_elements(),
        title="Class",
        loc="upper right",
    )
    axis.add_artist(legend)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_classification_grid(
    *,
    grid_limit: float,
    grid_points: int,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    """Create a square 2D evaluation grid."""
    if grid_limit <= 0:
        raise ValueError(
            "grid_limit must be positive."
        )
    if grid_points < 2:
        raise ValueError(
            "grid_points must be at least two."
        )

    axis_values = torch.linspace(
        -grid_limit,
        grid_limit,
        grid_points,
    )
    grid_x_1, grid_x_2 = torch.meshgrid(
        axis_values,
        axis_values,
        indexing="xy",
    )
    grid = torch.stack(
        [
            grid_x_1.reshape(-1),
            grid_x_2.reshape(-1),
        ],
        dim=1,
    )
    return (
        grid,
        grid_x_1.numpy(),
        grid_x_2.numpy(),
    )


def predict_logits_in_batches(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 65536,
) -> torch.Tensor:
    """Predict a large grid without requiring one large activation batch."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    batches = []
    for start in range(0, x.shape[0], batch_size):
        batches.append(
            predict_logits(
                model=model,
                x=x[start : start + batch_size],
            )
        )
    return torch.cat(
        batches,
        dim=0,
    )


def plot_decision_boundary(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    title: str,
    output_path: Path,
    grid_limit: float = 1.2,
    grid_points: int = 300,
) -> None:
    """Plot class regions together with observed samples."""
    grid, grid_x_1, grid_x_2 = (
        make_classification_grid(
            grid_limit=grid_limit,
            grid_points=grid_points,
        )
    )
    logits = predict_logits_in_batches(
        model=model,
        x=grid,
    )
    predictions = logits.argmax(
        dim=1
    ).reshape(
        grid_points,
        grid_points,
    )

    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    figure, axis = plt.subplots(
        figsize=(7.4, 7.0)
    )
    axis.contourf(
        grid_x_1,
        grid_x_2,
        predictions.numpy(),
        levels=np.arange(
            logits.shape[1] + 1
        ) - 0.5,
        alpha=0.28,
    )
    axis.scatter(
        x_numpy[:, 0],
        x_numpy[:, 1],
        c=y_numpy,
        s=13,
        alpha=0.85,
    )
    axis.set_xlim(
        -grid_limit,
        grid_limit,
    )
    axis.set_ylim(
        -grid_limit,
        grid_limit,
    )
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_logit_margin(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    title: str,
    output_path: Path,
    grid_limit: float = 1.2,
    grid_points: int = 300,
) -> None:
    """Plot the top-two logit margin over the input plane."""
    grid, grid_x_1, grid_x_2 = (
        make_classification_grid(
            grid_limit=grid_limit,
            grid_points=grid_points,
        )
    )
    logits = predict_logits_in_batches(
        model=model,
        x=grid,
    )
    top_two = torch.topk(
        logits,
        k=2,
        dim=1,
    ).values
    margin = (
        top_two[:, 0] - top_two[:, 1]
    ).reshape(
        grid_points,
        grid_points,
    )

    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    figure, axis = plt.subplots(
        figsize=(7.6, 7.0)
    )
    image = axis.contourf(
        grid_x_1,
        grid_x_2,
        margin.numpy(),
        levels=30,
    )
    axis.scatter(
        x_numpy[:, 0],
        x_numpy[:, 1],
        c=y_numpy,
        s=10,
        alpha=0.65,
    )
    axis.set_xlim(
        -grid_limit,
        grid_limit,
    )
    axis.set_ylim(
        -grid_limit,
        grid_limit,
    )
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(title)
    axis.set_aspect("equal")
    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(
        "Top-two logit margin"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_training_curves(
    losses: list[float],
    accuracies: list[float],
    *,
    loss_output_path: Path,
    accuracy_output_path: Path,
) -> None:
    """Plot full-batch training loss and accuracy."""
    steps = np.arange(
        1,
        len(losses) + 1,
    )

    loss_figure, loss_axis = plt.subplots(
        figsize=(8.5, 4.8)
    )
    loss_axis.plot(
        steps,
        losses,
    )
    loss_axis.set_yscale("log")
    loss_axis.set_xlabel("Training step")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title("Training loss")
    loss_axis.grid(alpha=0.25)
    loss_figure.tight_layout()
    loss_figure.savefig(
        loss_output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(loss_figure)

    accuracy_figure, accuracy_axis = plt.subplots(
        figsize=(8.5, 4.8)
    )
    accuracy_axis.plot(
        steps,
        accuracies,
    )
    accuracy_axis.set_xlabel("Training step")
    accuracy_axis.set_ylabel("Training accuracy")
    accuracy_axis.set_title("Training accuracy")
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.grid(alpha=0.25)
    accuracy_figure.tight_layout()
    accuracy_figure.savefig(
        accuracy_output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(accuracy_figure)


def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Plot a confusion matrix with true labels on rows."""
    matrix = confusion_matrix.detach().cpu().numpy()
    num_classes = matrix.shape[0]

    figure, axis = plt.subplots(
        figsize=(6.2, 5.5)
    )
    image = axis.imshow(matrix)
    figure.colorbar(
        image,
        ax=axis,
        label="Sample count",
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    axis.set_xticks(
        np.arange(num_classes)
    )
    axis.set_yticks(
        np.arange(num_classes)
    )

    for row in range(num_classes):
        for column in range(num_classes):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_checkpoint_boundaries(
    model: nn.Module,
    snapshots: dict[int, dict[str, torch.Tensor]],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    output_path: Path,
    grid_limit: float = 1.2,
    grid_points: int = 220,
) -> None:
    """Show how the classification regions develop during training."""
    steps = sorted(snapshots)
    columns = 2
    rows = math.ceil(
        len(steps) / columns
    )

    grid, grid_x_1, grid_x_2 = (
        make_classification_grid(
            grid_limit=grid_limit,
            grid_points=grid_points,
        )
    )
    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    current_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.5, 5.7 * rows),
    )
    axes_array = np.asarray(
        axes
    ).reshape(-1)

    try:
        for axis, step in zip(
            axes_array,
            steps,
        ):
            model.load_state_dict(
                snapshots[step]
            )
            logits = predict_logits_in_batches(
                model=model,
                x=grid,
            )
            predictions = logits.argmax(
                dim=1
            ).reshape(
                grid_points,
                grid_points,
            )
            axis.contourf(
                grid_x_1,
                grid_x_2,
                predictions.numpy(),
                levels=np.arange(
                    logits.shape[1] + 1
                ) - 0.5,
                        alpha=0.28,
            )
            axis.scatter(
                x_numpy[:, 0],
                x_numpy[:, 1],
                c=y_numpy,
                        s=8,
                        alpha=0.65,
            )
            axis.set_xlim(
                -grid_limit,
                grid_limit,
            )
            axis.set_ylim(
                -grid_limit,
                grid_limit,
            )
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")
            axis.set_title(
                f"Step {step}"
            )
            axis.set_aspect("equal")
    finally:
        model.load_state_dict(current_state)

    for unused_axis in axes_array[len(steps) :]:
        unused_axis.axis("off")

    figure.suptitle(
        "Decision boundary during training",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=175,
        bbox_inches="tight",
    )
    plt.close(figure)
