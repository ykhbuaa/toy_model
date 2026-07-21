import math

import pytest
import torch

from toy_model.data import sample_spiral


def test_spiral_shapes_and_balanced_classes() -> None:
    x, y = sample_spiral(
        num_classes=3,
        samples_per_class=40,
        seed=7,
    )

    assert x.shape == (120, 2)
    assert y.shape == (120,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long

    counts = torch.bincount(
        y,
        minlength=3,
    )
    assert torch.equal(
        counts,
        torch.tensor([40, 40, 40]),
    )


def test_spiral_sampling_is_reproducible() -> None:
    x_1, y_1 = sample_spiral(
        samples_per_class=32,
        seed=42,
    )
    x_2, y_2 = sample_spiral(
        samples_per_class=32,
        seed=42,
    )

    assert torch.equal(x_1, x_2)
    assert torch.equal(y_1, y_2)


def test_spiral_respects_radial_bounds_without_noise() -> None:
    radius_min = 0.2
    radius_max = 0.9

    x, _ = sample_spiral(
        samples_per_class=200,
        radius_min=radius_min,
        radius_max=radius_max,
        angular_noise_std=0.0,
        seed=1,
        shuffle=False,
    )
    radius = torch.linalg.vector_norm(
        x,
        dim=1,
    )

    assert float(radius.min()) >= radius_min - 1e-6
    assert float(radius.max()) <= radius_max + 1e-6


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("num_classes", 1),
        ("samples_per_class", 0),
        ("turns", 0.0),
        ("radius_min", -0.1),
        ("angular_noise_std", -0.1),
    ],
)
def test_spiral_rejects_invalid_parameters(
    argument: str,
    value: float | int,
) -> None:
    kwargs = {argument: value}
    with pytest.raises(ValueError):
        sample_spiral(**kwargs)


def test_spiral_rejects_invalid_radius_interval() -> None:
    with pytest.raises(ValueError):
        sample_spiral(
            radius_min=1.0,
            radius_max=1.0,
        )
