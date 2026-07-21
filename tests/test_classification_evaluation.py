import math

import torch

from toy_model.evaluation import (
    compute_confusion_matrix,
    compute_logit_margins,
    evaluate_classification_logits,
)


def make_logits_and_targets() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )
    return logits, targets


def test_classification_metrics_are_correct() -> None:
    logits, targets = make_logits_and_targets()
    metrics = evaluate_classification_logits(
        logits=logits,
        targets=targets,
    )

    assert metrics.accuracy == 0.75
    assert metrics.per_class_accuracy == (
        0.5,
        1.0,
        1.0,
    )
    assert math.isclose(
        metrics.mean_top_two_margin,
        1.5,
    )
    assert math.isclose(
        metrics.mean_true_class_margin,
        1.0,
    )
    assert metrics.cross_entropy > 0.0
    assert (
        0.0
        < metrics.mean_correct_class_probability
        < 1.0
    )


def test_confusion_matrix_uses_true_rows() -> None:
    logits, targets = make_logits_and_targets()
    confusion = compute_confusion_matrix(
        logits=logits,
        targets=targets,
    )

    expected = torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    assert torch.equal(confusion, expected)


def test_true_class_margin_is_negative_for_error() -> None:
    logits, targets = make_logits_and_targets()
    _, true_class_margin = compute_logit_margins(
        logits=logits,
        targets=targets,
    )

    assert true_class_margin[-1] == -1.0
    assert bool((true_class_margin[:3] > 0).all())
