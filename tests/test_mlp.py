import pytest
import torch

from toy_model.models import MLP


def test_mlp_output_shape() -> None:
    model = MLP(
        input_dim=1,
        hidden_dims=[16, 32],
        output_dim=1,
        activation="relu",
    )

    x = torch.randn(8, 1)
    y = model(x)

    assert y.shape == (8, 1)


def test_mlp_has_trainable_parameters() -> None:
    model = MLP(
        input_dim=2,
        hidden_dims=[8, 8],
        output_dim=3,
    )

    assert model.count_parameters() > 0


def test_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError):
        MLP(
            input_dim=1,
            hidden_dims=[16],
            output_dim=1,
            activation="unknown",
        )


def test_mlp_rejects_empty_hidden_dims() -> None:
    with pytest.raises(ValueError):
        MLP(
            input_dim=1,
            hidden_dims=[],
            output_dim=1,
        )
