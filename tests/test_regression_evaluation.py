import torch
from torch import nn

from toy_model.evaluation import evaluate_function_regression


def test_exact_linear_model_has_near_zero_error() -> None:
    model = nn.Linear(
        in_features=1,
        out_features=1,
    )

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(1.0)

    x_train = torch.linspace(
        -1.0,
        1.0,
        32,
    ).unsqueeze(1)

    y_train = 2.0 * x_train + 1.0

    metrics, x_grid, y_true, prediction = (
        evaluate_function_regression(
            model=model,
            x_train=x_train,
            y_train=y_train,
            function_name="linear",
            train_min=-1.0,
            train_max=1.0,
            evaluation_min=-2.0,
            evaluation_max=2.0,
            grid_points=512,
        )
    )

    assert x_grid.shape == (512, 1)
    assert y_true.shape == (512, 1)
    assert prediction.shape == (512, 1)

    assert metrics.train_label_mse < 1e-12
    assert metrics.train_clean_mse < 1e-12
    assert metrics.interpolation_mse < 1e-12
    assert metrics.extrapolation_mse < 1e-12
