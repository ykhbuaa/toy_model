import torch
from torch import nn

from toy_model.models import MLP
from toy_model.training import (
    RegressionTrainConfig,
    fit_regression,
)


def test_regression_training_reduces_loss() -> None:
    torch.manual_seed(42)

    x = torch.linspace(
        -1.0,
        1.0,
        128,
    ).unsqueeze(1)

    y = 2.0 * x + 1.0

    model = MLP(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1,
        activation="relu",
    )

    loss_function = nn.MSELoss()

    with torch.no_grad():
        initial_loss = float(
            loss_function(
                model(x),
                y,
            ).item()
        )

    config = RegressionTrainConfig(
        steps=400,
        learning_rate=1e-2,
        device="cpu",
        log_every=0,
        checkpoint_steps=(0, 400),
    )

    result = fit_regression(
        model=model,
        x_train=x,
        y_train=y,
        config=config,
        verbose=False,
    )

    assert result.final_loss < initial_loss
    assert result.final_loss < 0.01
    assert 0 in result.snapshots
    assert 400 in result.snapshots
