"""Synthetic two-dimensional classification datasets."""

from __future__ import annotations

import math

import torch


def sample_spiral(
    *,
    num_classes: int = 3,
    samples_per_class: int = 300,
    turns: float = 1.5,
    radius_min: float = 0.1,
    radius_max: float = 1.0,
    angular_noise_std: float = 0.15,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sample a balanced multi-class spiral dataset.

    Each class is a rotated copy of the same Archimedean-style spiral.
    A latent progress variable ``u`` controls both radius and angle:

    .. math::

        r = r_{\min} + (r_{\max} - r_{\min})u

    .. math::

        \theta = \frac{2\pi c}{C} + 2\pi T u + \epsilon_\theta

    where ``c`` is the class index, ``C`` is the number of classes,
    ``T`` is the number of turns, and angular noise is Gaussian.

    ``radius_min`` is positive by default so the classes do not collapse
    into an intrinsically ambiguous point at the origin. That keeps the
    first experiment focused on decision-boundary geometry rather than
    irreducible label ambiguity.

    Args:
        num_classes: Number of balanced classes. Must be at least two.
        samples_per_class: Number of samples generated for every class.
        turns: Number of spiral revolutions over the radial interval.
        radius_min: Smallest radius.
        radius_max: Largest radius.
        angular_noise_std: Standard deviation of angular noise in radians.
        seed: Optional seed local to this sampling call.
        shuffle: Whether to randomly permute all classes after generation.

    Returns:
        ``(x, y)`` where ``x`` has shape ``[N, 2]`` and dtype
        ``torch.float32``, while ``y`` has shape ``[N]`` and dtype
        ``torch.long``.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    if turns <= 0:
        raise ValueError("turns must be positive.")
    if radius_min < 0:
        raise ValueError("radius_min cannot be negative.")
    if radius_min >= radius_max:
        raise ValueError("radius_min must be smaller than radius_max.")
    if angular_noise_std < 0:
        raise ValueError("angular_noise_std cannot be negative.")

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for class_index in range(num_classes):
        progress = torch.rand(
            samples_per_class,
            generator=generator,
            dtype=torch.float32,
        )
        radius = (
            radius_min
            + (radius_max - radius_min) * progress
        )
        class_phase = (
            2.0
            * math.pi
            * class_index
            / num_classes
        )
        angle = (
            class_phase
            + 2.0 * math.pi * turns * progress
        )

        if angular_noise_std > 0:
            angular_noise = torch.randn(
                samples_per_class,
                generator=generator,
                dtype=torch.float32,
            )
            angle = angle + angular_noise_std * angular_noise

        class_features = torch.stack(
            [
                radius * torch.cos(angle),
                radius * torch.sin(angle),
            ],
            dim=1,
        )
        class_labels = torch.full(
            (samples_per_class,),
            fill_value=class_index,
            dtype=torch.long,
        )
        features.append(class_features)
        labels.append(class_labels)

    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    if shuffle:
        permutation = torch.randperm(
            x.shape[0],
            generator=generator,
        )
        x = x[permutation]
        y = y[permutation]

    return x, y
