"""Evaluation utilities for supervised classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from toy_model.training.classification import predict_logits


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics describing classification quality and confidence."""

    cross_entropy: float
    accuracy: float
    per_class_accuracy: tuple[float, ...]
    mean_top_two_margin: float
    mean_true_class_margin: float
    mean_correct_class_probability: float

    def to_dict(
        self,
        *,
        prefix: str = "",
    ) -> dict[str, float]:
        """Convert metrics to flat, CSV-friendly key-value pairs."""
        result = {
            f"{prefix}cross_entropy": self.cross_entropy,
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}mean_top_two_margin": (
                self.mean_top_two_margin
            ),
            f"{prefix}mean_true_class_margin": (
                self.mean_true_class_margin
            ),
            f"{prefix}mean_correct_class_probability": (
                self.mean_correct_class_probability
            ),
        }
        for class_index, class_accuracy in enumerate(
            self.per_class_accuracy
        ):
            result[
                f"{prefix}class_{class_index}_accuracy"
            ] = class_accuracy
        return result


def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [num_samples, num_classes]."
        )
    if logits.shape[1] < 2:
        raise ValueError(
            "At least two classes are required."
        )
    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [num_samples]."
        )
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "logits and targets must contain the same samples."
        )
    if logits.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if targets.dtype != torch.long:
        raise ValueError(
            "targets must have dtype torch.long."
        )
    if int(targets.min().item()) < 0:
        raise ValueError("targets cannot be negative.")
    if int(targets.max().item()) >= logits.shape[1]:
        raise ValueError(
            "A target label is outside the logits range."
        )


def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute a confusion matrix with rows=true and columns=predicted."""
    _validate_logits_and_targets(
        logits=logits,
        targets=targets,
    )
    num_classes = logits.shape[1]
    predictions = logits.argmax(dim=1)
    flattened_indices = (
        targets * num_classes + predictions
    )
    counts = torch.bincount(
        flattened_indices,
        minlength=num_classes * num_classes,
    )
    return counts.reshape(
        num_classes,
        num_classes,
    )


def compute_logit_margins(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-two confidence margins and signed true-class margins.

    The top-two margin is ``largest_logit - second_largest_logit`` and is
    always non-negative. The true-class margin is
    ``true_logit - largest_incorrect_logit``; it is negative exactly when
    another class has a larger logit than the target class.
    """
    _validate_logits_and_targets(
        logits=logits,
        targets=targets,
    )

    top_two_values = torch.topk(
        logits,
        k=2,
        dim=1,
    ).values
    top_two_margin = (
        top_two_values[:, 0] - top_two_values[:, 1]
    )

    true_logits = logits.gather(
        dim=1,
        index=targets.unsqueeze(1),
    ).squeeze(1)
    incorrect_logits = logits.clone()
    incorrect_logits.scatter_(
        dim=1,
        index=targets.unsqueeze(1),
        value=-torch.inf,
    )
    largest_incorrect_logits = incorrect_logits.max(
        dim=1
    ).values
    true_class_margin = (
        true_logits - largest_incorrect_logits
    )

    return top_two_margin, true_class_margin


def evaluate_classification_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> ClassificationMetrics:
    """Evaluate already-computed logits against integer class labels."""
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()
    _validate_logits_and_targets(
        logits=logits,
        targets=targets,
    )

    predictions = logits.argmax(dim=1)
    accuracy = float(
        (predictions == targets).float().mean().item()
    )

    per_class_accuracy: list[float] = []
    for class_index in range(logits.shape[1]):
        class_mask = targets == class_index
        if bool(class_mask.any()):
            value = (
                predictions[class_mask]
                == targets[class_mask]
            ).float().mean()
            per_class_accuracy.append(
                float(value.item())
            )
        else:
            per_class_accuracy.append(float("nan"))

    top_two_margin, true_class_margin = (
        compute_logit_margins(
            logits=logits,
            targets=targets,
        )
    )
    probabilities = torch.softmax(
        logits,
        dim=1,
    )
    correct_class_probabilities = probabilities.gather(
        dim=1,
        index=targets.unsqueeze(1),
    ).squeeze(1)

    return ClassificationMetrics(
        cross_entropy=float(
            F.cross_entropy(
                logits,
                targets,
            ).item()
        ),
        accuracy=accuracy,
        per_class_accuracy=tuple(
            per_class_accuracy
        ),
        mean_top_two_margin=float(
            top_two_margin.mean().item()
        ),
        mean_true_class_margin=float(
            true_class_margin.mean().item()
        ),
        mean_correct_class_probability=float(
            correct_class_probabilities.mean().item()
        ),
    )


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[
    ClassificationMetrics,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate a classifier and return metrics, logits and confusion."""
    logits = predict_logits(
        model=model,
        x=x,
    )
    targets_cpu = targets.detach().cpu()
    metrics = evaluate_classification_logits(
        logits=logits,
        targets=targets_cpu,
    )
    confusion_matrix = compute_confusion_matrix(
        logits=logits,
        targets=targets_cpu,
    )
    return metrics, logits, confusion_matrix
