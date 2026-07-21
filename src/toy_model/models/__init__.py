"""Neural network models."""

from toy_model.models.mlp import MLP, build_activation

from toy_model.models.fourier import (
    FourierFeatureEncoding,
    FourierFeatureMLP,
)

__all__ = [
    "MLP",
    "build_activation",
    "FourierFeatureEncoding",
    "FourierFeatureMLP"
]
