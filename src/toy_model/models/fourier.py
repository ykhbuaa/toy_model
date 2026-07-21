"""Deterministic Fourier-feature models for periodic one-dimensional signals."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from toy_model.models.mlp import MLP


class FourierFeatureEncoding(nn.Module):
    """Map a scalar coordinate to sinusoidal Fourier features.

    For frequencies [w_1, ..., w_k], the encoding is:

        [
            sin(w_1 x), ..., sin(w_k x),
            cos(w_1 x), ..., cos(w_k x),
        ]

    When all frequencies are integers and ``include_input=False``,
    the complete encoding is exactly 2*pi periodic.

    Args:
        frequencies:
            Positive angular frequencies.
        include_input:
            Whether to also include the original coordinate x.
            Setting this to False preserves exact periodicity when the
            frequencies are integers.
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        *,
        include_input: bool = False,
    ) -> None:
        super().__init__()

        if not frequencies:
            raise ValueError("frequencies cannot be empty.")

        frequency_tensor = torch.tensor(
            list(frequencies),
            dtype=torch.float32,
        )

        if frequency_tensor.ndim != 1:
            raise ValueError(
                "frequencies must be one-dimensional."
            )

        if not bool(torch.all(frequency_tensor > 0)):
            raise ValueError(
                "All frequencies must be positive."
            )

        self.register_buffer(
            "frequencies",
            frequency_tensor.reshape(1, -1),
        )

        self.include_input = include_input

    @property
    def output_dim(self) -> int:
        """Return the number of encoded features."""
        sinusoidal_dim = 2 * self.frequencies.shape[1]

        return sinusoidal_dim + int(self.include_input)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode coordinates with sinusoidal features."""
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(
                "FourierFeatureEncoding expects x with shape [N, 1]."
            )

        angles = x * self.frequencies

        features = [
            torch.sin(angles),
            torch.cos(angles),
        ]

        if self.include_input:
            features.insert(0, x)

        return torch.cat(
            features,
            dim=1,
        )


class FourierFeatureMLP(nn.Module):
    """An MLP operating on deterministic Fourier features.

    Args:
        frequencies:
            Frequencies used by the Fourier encoding.
        hidden_dims:
            Width of each hidden layer.
        output_dim:
            Number of output values.
        activation:
            Activation inside the MLP.
        include_input:
            Whether to concatenate raw x to the Fourier features.
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        hidden_dims: Sequence[int],
        output_dim: int = 1,
        activation: str = "relu",
        *,
        include_input: bool = False,
    ) -> None:
        super().__init__()

        self.encoding = FourierFeatureEncoding(
            frequencies=frequencies,
            include_input=include_input,
        )

        self.mlp = MLP(
            input_dim=self.encoding.output_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode x and predict the output."""
        encoded_x = self.encoding(x)

        return self.mlp(encoded_x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
