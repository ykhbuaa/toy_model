"""Synthetic datasets used by toy_model experiments."""

from toy_model.data.classification_2d import sample_spiral
from toy_model.data.functions_1d import (
    evaluate_function,
    sample_function,
    sample_function_grid,
)

__all__ = [
    "evaluate_function",
    "sample_function",
    "sample_function_grid",
    "sample_spiral"
]
