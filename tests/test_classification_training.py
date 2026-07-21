import pytest
import torch
from torch import nn

from toy_model.training import (
    ClassificationTrainConfig,
    fit_classification,
)


def test_classification_training_learns_separable_data() -> None:
    torch.manual_seed(42)

    negative = torch.randn(64, 2) * 0.15
    negative[:, 0] -= 1.0
    positive = torch.randn(64, 2) * 0.15
    positive[:, 0] += 1.0

    x = torch.cat(
        [negative, positive],
        dim=0,
    )
    y = torch.cat(
        [
            torch.zeros(64, dtype=torch.long),
            torch.ones(64, dtype=torch.long),
        ],
        dim=0,
    )

    model = nn.Linear(2, 2)
    initial_logits = model(x)
    initial_loss = float(
        nn.CrossEntropyLoss()(
            initial_logits,
            y,
        ).item()
    )

    config = ClassificationTrainConfig(
        steps=60,
        learning_rate=5e-2,
        device="cpu",
        log_every=0,
        checkpoint_steps=(0, 60),
    )
    result = fit_classification(
        model=model,
        x_train=x,
        y_train=y,
        config=config,
        verbose=False,
    )

    assert result.final_loss < initial_loss
    assert result.final_accuracy > 0.99
    assert 0 in result.snapshots
    assert 60 in result.snapshots


def test_classification_requires_long_labels() -> None:
    model = nn.Linear(2, 2)
    x = torch.randn(8, 2)
    y = torch.zeros(8)

    with pytest.raises(ValueError):
        fit_classification(
            model=model,
            x_train=x,
            y_train=y,
            config=ClassificationTrainConfig(
                steps=1,
                device="cpu",
                log_every=0,
            ),
            verbose=False,
        )


def test_classification_detects_output_label_mismatch() -> None:
    model = nn.Linear(2, 2)
    x = torch.randn(8, 2)
    y = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        fit_classification(
            model=model,
            x_train=x,
            y_train=y,
            config=ClassificationTrainConfig(
                steps=1,
                device="cpu",
                log_every=0,
            ),
            verbose=False,
        )
