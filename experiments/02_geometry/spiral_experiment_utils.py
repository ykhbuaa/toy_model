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

from toy_model.diagnostics import record_activation_stats
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


def compute_input_gradient_norms(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``||d selected_logit / d x||`` and selected classes.

    The selected logit is the model's predicted class for each input. Since
    each sample is independent in a feed-forward classifier, differentiating
    the sum of selected logits gives one input gradient per sample.
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = x.detach().to(device).requires_grad_(True)
    logits = model(inputs)
    predicted_classes = logits.argmax(dim=1)
    selected_logits = logits.gather(
        dim=1,
        index=predicted_classes.unsqueeze(1),
    ).squeeze(1)
    gradients = torch.autograd.grad(
        selected_logits.sum(),
        inputs,
    )[0]
    return (
        gradients.detach().norm(dim=1).cpu(),
        predicted_classes.detach().cpu(),
    )


def compute_boundary_complexity(
    predictions: torch.Tensor,
    *,
    grid_points: int,
) -> float:
    """Estimate boundary complexity by normalized grid label transitions."""
    if predictions.ndim != 1:
        raise ValueError("predictions must be a flat grid vector.")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two.")
    expected = grid_points * grid_points
    if predictions.numel() != expected:
        raise ValueError(
            f"Expected {expected} predictions, got {predictions.numel()}."
        )
    labels = predictions.reshape(grid_points, grid_points)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    transition_count = int(horizontal.sum().item() + vertical.sum().item())
    possible_count = horizontal.numel() + vertical.numel()
    return transition_count / possible_count


def collect_layer_diagnostics(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect activation summaries and two-dimensional PCA summaries.

    The probe set is fixed by the caller. Hidden vectors are retained only
    for this small diagnostic call, then converted into scalar CSV rows.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    recorder = record_activation_stats(
        model,
        x.to(device),
        store_samples=True,
    )
    activation_rows = [stat.to_dict() for stat in recorder.stats()]
    targets_cpu = targets.detach().cpu()
    pca_rows: list[dict[str, object]] = []
    for layer_name, values in recorder.samples().items():
        features = values.reshape(values.shape[0], -1).float()
        centered = features - features.mean(dim=0, keepdim=True)
        if features.shape[0] > 1:
            singular_values = torch.linalg.svdvals(centered)
            variances = singular_values.square() / (features.shape[0] - 1)
            total_variance = float(variances.sum().item())
            explained = (
                variances / variances.sum()
                if total_variance > 0
                else torch.zeros_like(variances)
            )
            components = min(2, singular_values.numel())
            projection = centered @ torch.linalg.svd(centered, full_matrices=False).Vh[:components].T
        else:
            explained = torch.zeros(1)
            projection = torch.zeros(features.shape[0], 1)
            total_variance = 0.0

        global_mean = features.mean(dim=0)
        within_distances: list[float] = []
        class_means: list[torch.Tensor] = []
        for class_index in torch.unique(targets_cpu, sorted=True):
            class_values = features[targets_cpu == class_index]
            class_mean = class_values.mean(dim=0)
            class_means.append(class_mean)
            if class_values.shape[0] > 1:
                within_distances.append(
                    float((class_values - class_mean).norm(dim=1).mean().item())
                )
        within = float(np.mean(within_distances)) if within_distances else 0.0
        between = 0.0
        if class_means:
            between = float(torch.stack(class_means).sub(global_mean).norm(dim=1).mean().item())
        pca_rows.append(
            {
                "module_name": layer_name,
                "feature_dim": int(features.shape[1]),
                "total_variance": total_variance,
                "effective_rank": int(
                    (explained > 1e-6).sum().item()
                ),
                "within_class_distance": within,
                "between_class_distance": between,
                "between_within_ratio": between / within if within > 0 else float("nan"),
                "pc1_explained_variance": float(explained[0].item()),
                "pc2_explained_variance": float(explained[1].item()) if explained.numel() > 1 else 0.0,
                "projection": projection.numpy(),
                "targets": targets_cpu.numpy(),
            }
        )
    return activation_rows, pca_rows


def plot_input_gradient_norm(
    grid_x_1: np.ndarray,
    grid_x_2: np.ndarray,
    gradient_norms: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Plot the predicted-logit input sensitivity over a 2D grid."""
    values = gradient_norms.detach().cpu().reshape(grid_x_1.shape).numpy()
    figure, axis = plt.subplots(figsize=(7.6, 7.0))
    image = axis.contourf(grid_x_1, grid_x_2, values, levels=30)
    axis.scatter(
        x.detach().cpu().numpy()[:, 0],
        x.detach().cpu().numpy()[:, 1],
        c=y.detach().cpu().numpy(),
        s=10,
        alpha=0.65,
    )
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(title)
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="Input gradient norm")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_layer_pca(
    pca_rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot the first two activation PCA coordinates for each hidden layer."""
    if not pca_rows:
        return
    columns = 2
    rows = math.ceil(len(pca_rows) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11.0, 4.7 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, row in zip(axes_array, pca_rows):
        projection = np.asarray(row["projection"])
        targets = np.asarray(row["targets"])
        axis.scatter(
            projection[:, 0],
            projection[:, 1] if projection.shape[1] > 1 else np.zeros(projection.shape[0]),
            c=targets,
            s=13,
            alpha=0.75,
        )
        axis.set_title(
            f"{row['module_name']} | "
            f"PC1={float(row['pc1_explained_variance']):.2f}, "
            f"PC2={float(row['pc2_explained_variance']):.2f}"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.2)
    for axis in axes_array[len(pca_rows):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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
