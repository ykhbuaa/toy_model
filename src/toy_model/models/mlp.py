"""Multilayer perceptron models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    """Build an activation module from its name."""
    normalized_name = name.lower()

    if normalized_name == "relu":
        return nn.ReLU()

    if normalized_name == "tanh":
        return nn.Tanh()

    if normalized_name == "silu":
        return nn.SiLU()

    if normalized_name == "gelu":
        return nn.GELU()

    raise ValueError(
        f"Unknown activation: {name!r}. "
        "Expected one of: relu, tanh, silu, gelu."
    )


class MLP(nn.Module):
    """A configurable fully connected neural network.

    The final layer does not apply an activation function, which is suitable
    for ordinary regression tasks.

    Args:
        input_dim:
            Number of input features.
        hidden_dims:
            Width of each hidden layer.
        output_dim:
            Number of output features.
        activation:
            Activation used after every hidden linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty.")

        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive.")

        dimensions = [
            input_dim,
            *hidden_dims,
            output_dim,
        ]

        layers: list[nn.Module] = []

        for layer_index in range(len(dimensions) - 1):
            input_features = dimensions[layer_index]
            output_features = dimensions[layer_index + 1]

            layers.append(
                nn.Linear(
                    input_features,
                    output_features,
                )
            )

            is_output_layer = layer_index == len(dimensions) - 2

            if not is_output_layer:
                layers.append(build_activation(activation))

        self.network = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute model predictions."""
        return self.network(x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
