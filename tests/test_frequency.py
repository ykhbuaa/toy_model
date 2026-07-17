import math

import torch

from toy_model.analysis import (
    compute_amplitude_spectrum,
    find_nearest_frequency_amplitude,
)


def test_mixed_sine_frequency_amplitudes() -> None:
    num_samples = 4096
    x_min = -math.pi
    x_max = math.pi

    spacing = (
        x_max - x_min
    ) / num_samples

    x = (
        x_min
        + torch.arange(num_samples)
        * spacing
    ).unsqueeze(1)

    y = (
        torch.sin(x)
        + 0.3 * torch.sin(10.0 * x)
    )

    frequencies, amplitudes = compute_amplitude_spectrum(
        x=x,
        y=y,
    )

    low_frequency_amplitude = (
        find_nearest_frequency_amplitude(
            angular_frequencies=frequencies,
            amplitudes=amplitudes,
            target_frequency=1.0,
        )
    )

    high_frequency_amplitude = (
        find_nearest_frequency_amplitude(
            angular_frequencies=frequencies,
            amplitudes=amplitudes,
            target_frequency=10.0,
        )
    )

    assert abs(low_frequency_amplitude - 1.0) < 0.01
    assert abs(high_frequency_amplitude - 0.3) < 0.01
