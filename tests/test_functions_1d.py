import math

import torch

from toy_model.data import evaluate_function, sample_function


def test_linear_function() -> None:
    x = torch.tensor([[0.0], [1.0], [-1.0]])

    y = evaluate_function(x, "linear")

    expected = torch.tensor([[1.0], [3.0], [-1.0]])

    assert torch.allclose(y, expected)


def test_sine_function() -> None:
    x = torch.tensor(
        [
            [0.0],
            [math.pi / 2.0],
            [math.pi],
        ]
    )

    y = evaluate_function(x, "sine")

    expected = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    assert torch.allclose(y, expected, atol=1e-6)


def test_sample_shapes() -> None:
    x, y = sample_function(
        function_name="mixed_sine",
        num_samples=32,
        seed=42,
    )

    assert x.shape == (32, 1)
    assert y.shape == (32, 1)


def test_sampling_is_reproducible() -> None:
    x1, y1 = sample_function(
        function_name="sine",
        num_samples=16,
        seed=42,
    )

    x2, y2 = sample_function(
        function_name="sine",
        num_samples=16,
        seed=42,
    )

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
