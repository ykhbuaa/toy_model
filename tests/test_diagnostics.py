import torch

from toy_model.diagnostics import (
    capture_parameter_state,
    gradient_norms,
    parameter_norms,
    record_activation_stats,
    update_norms,
)
from toy_model.models import MLP


def test_activation_recorder_collects_relu_statistics_without_changing_output() -> None:
    model = MLP(
        input_dim=2,
        hidden_dims=[4, 3],
        output_dim=2,
        activation="relu",
    )
    inputs = torch.tensor(
        [[-1.0, 0.5], [0.2, 1.0], [1.0, -0.3]],
    )
    expected = model(inputs).detach()

    recorder = record_activation_stats(
        model,
        inputs,
        store_samples=True,
    )

    assert torch.allclose(model(inputs), expected)
    assert len(recorder.stats()) == 2
    assert all(0.0 <= stat.zero_fraction <= 1.0 for stat in recorder.stats())
    assert set(recorder.samples()) == {
        "network.1",
        "network.3",
    }


def test_parameter_and_gradient_norms_are_finite_and_updates_are_measurable() -> None:
    model = MLP(
        input_dim=2,
        hidden_dims=[4],
        output_dim=2,
        activation="relu",
    )
    inputs = torch.randn(5, 2)
    targets = torch.tensor([0, 1, 0, 1, 0])
    before = capture_parameter_state(model)
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()

    norms = parameter_norms(model)
    gradients = gradient_norms(model)
    assert set(norms) == set(gradients) == set(before)
    assert all(torch.isfinite(torch.tensor(value)) for value in norms.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in gradients.values())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    updates = update_norms(model, before)
    assert any(value > 0.0 for value in updates.values())
