"""Training utilities for supervised regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RegressionTrainConfig:
    """Configuration for a simple full-batch regression experiment."""

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
            raise ValueError("checkpoint_steps cannot contain negative values.")


@dataclass
class RegressionTrainResult:
    """Results produced by a regression training run."""

    losses: list[float]
    snapshots: dict[int, dict[str, torch.Tensor]]
    device: str

    @property
    def final_loss(self) -> float:
        """Return the final recorded training loss."""
        return self.losses[-1]


def resolve_device(requested_device: str) -> torch.device:
    """Resolve a user-facing device string to a PyTorch device."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, "
            "but CUDA is not available."
        )

    return device


def snapshot_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Copy a model state dictionary to CPU memory."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def fit_regression(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: RegressionTrainConfig,
    *,
    verbose: bool = True,
) -> RegressionTrainResult:
    """Train a model using full-batch mean squared error.

    Full-batch training is intentional for the first toy experiments. It
    removes mini-batch sampling noise, making the learning process easier to
    inspect.

    Args:
        model:
            PyTorch model to optimize.
        x_train:
            Input tensor with shape [num_samples, input_dim].
        y_train:
            Target tensor with shape [num_samples, output_dim].
        config:
            Training configuration.
        verbose:
            Whether to print training progress.

    Returns:
        Loss history, selected model snapshots and the resolved device.
    """
    if x_train.ndim != 2:
        raise ValueError(
            f"x_train must have two dimensions, got {x_train.shape}."
        )

    if y_train.ndim != 2:
        raise ValueError(
            f"y_train must have two dimensions, got {y_train.shape}."
        )

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "x_train and y_train must contain the same number of samples."
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

    loss_function = nn.MSELoss()

    losses: list[float] = []
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

        predictions = model(x_train)
        loss = loss_function(predictions, y_train)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Training produced a non-finite loss at step {step}: "
                f"{loss.item()}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)

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
                f"device={device}"
            )

    # Always preserve the final model, even if the user did not explicitly
    # include the final step in checkpoint_steps.
    if config.steps not in snapshots:
        snapshots[config.steps] = snapshot_state_dict(model)

    return RegressionTrainResult(
        losses=losses,
        snapshots=snapshots,
        device=str(device),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run inference and return predictions on CPU."""
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    predictions = model(x.to(device))

    return predictions.detach().cpu()


@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Evaluate mean squared error."""
    predictions = predict(model, x)

    return float(
        torch.mean(
            (predictions - y.detach().cpu()) ** 2
        ).item()
    )
