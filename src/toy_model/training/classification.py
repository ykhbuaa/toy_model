"""Training utilities for supervised classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from toy_model.training.regression import (
    resolve_device,
    snapshot_state_dict,
)


@dataclass(frozen=True)
class ClassificationTrainConfig:
    """Configuration for a simple full-batch classification experiment."""

    steps: int = 5000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    log_every: int = 500
    checkpoint_steps: tuple[int, ...] = (
        0,
        10,
        100,
        500,
        1000,
        5000,
    )

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.log_every < 0:
            raise ValueError("log_every cannot be negative.")
        if any(step < 0 for step in self.checkpoint_steps):
            raise ValueError(
                "checkpoint_steps cannot contain negative values."
            )


@dataclass
class ClassificationTrainResult:
    """Results produced by a full-batch classification training run."""

    losses: list[float]
    accuracies: list[float]
    snapshots: dict[int, dict[str, torch.Tensor]]
    device: str

    @property
    def final_loss(self) -> float:
        """Return the final recorded cross-entropy loss."""
        return self.losses[-1]

    @property
    def final_accuracy(self) -> float:
        """Return the final recorded training accuracy."""
        return self.accuracies[-1]


def _validate_classification_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError(
            f"x must have two dimensions, got {x.shape}."
        )
    if y.ndim != 1:
        raise ValueError(
            f"y must have one dimension, got {y.shape}."
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must contain the same number of samples."
        )
    if x.shape[0] == 0:
        raise ValueError("The classification dataset cannot be empty.")
    if y.dtype != torch.long:
        raise ValueError(
            "Classification labels must have dtype torch.long."
        )
    if int(y.min().item()) < 0:
        raise ValueError(
            "Classification labels cannot be negative."
        )


def fit_classification(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: ClassificationTrainConfig,
    *,
    verbose: bool = True,
) -> ClassificationTrainResult:
    """Train a classifier using full-batch cross entropy.

    The model must return unnormalized logits with shape
    ``[num_samples, num_classes]``. It should not apply softmax itself:
    ``torch.nn.CrossEntropyLoss`` combines a numerically stable
    log-softmax operation with negative log likelihood.

    Full-batch optimization is intentional in this first geometry
    experiment. It removes mini-batch sampling noise, so differences
    between model depths are easier to interpret.
    """
    _validate_classification_tensors(
        x=x_train,
        y=y_train,
    )

    device = resolve_device(config.device)
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    losses: list[float] = []
    accuracies: list[float] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}

    requested_snapshots = {
        step
        for step in config.checkpoint_steps
        if step <= config.steps
    }
    if 0 in requested_snapshots:
        snapshots[0] = snapshot_state_dict(model)

    for step in range(1, config.steps + 1):
        model.train()
        logits = model(x_train)

        if logits.ndim != 2:
            raise ValueError(
                "The classifier must return logits with shape [N, C]."
            )
        if logits.shape[0] != x_train.shape[0]:
            raise ValueError(
                "The classifier returned the wrong number of samples."
            )
        if logits.shape[1] < 2:
            raise ValueError(
                "The classifier must return at least two logits."
            )
        if int(y_train.max().item()) >= logits.shape[1]:
            raise ValueError(
                "A target label is outside the model output range."
            )
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                f"Training produced non-finite logits at step {step}."
            )

        loss = loss_function(logits, y_train)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training produced a non-finite loss at "
                f"step {step}: {loss.item()}."
            )

        predictions = logits.argmax(dim=1)
        accuracy = (
            predictions == y_train
        ).float().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        accuracy_value = float(
            accuracy.detach().cpu().item()
        )
        losses.append(loss_value)
        accuracies.append(accuracy_value)

        if step in requested_snapshots:
            snapshots[step] = snapshot_state_dict(model)

        should_log = (
            verbose
            and config.log_every > 0
            and (
                step == 1
                or step % config.log_every == 0
                or step == config.steps
            )
        )
        if should_log:
            print(
                f"step={step:6d} | "
                f"loss={loss_value:.8f} | "
                f"accuracy={accuracy_value:.4f} | "
                f"device={device}"
            )

    if config.steps not in snapshots:
        snapshots[config.steps] = snapshot_state_dict(model)

    return ClassificationTrainResult(
        losses=losses,
        accuracies=accuracies,
        snapshots=snapshots,
        device=str(device),
    )


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run classifier inference and return logits on CPU."""
    if x.ndim != 2:
        raise ValueError(
            f"x must have two dimensions, got {x.shape}."
        )

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    logits = model(x.to(device))
    if logits.ndim != 2:
        raise ValueError(
            "The classifier must return logits with shape [N, C]."
        )
    return logits.detach().cpu()


@torch.no_grad()
def predict_labels(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return the most likely class index for every input."""
    return predict_logits(
        model=model,
        x=x,
    ).argmax(dim=1)
