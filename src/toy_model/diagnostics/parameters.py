"""Parameter and gradient measurements for small transparent experiments."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def parameter_norms(model: nn.Module) -> dict[str, float]:
    """Return the L2 norm of every trainable parameter tensor."""
    return {
        name: float(parameter.detach().norm().cpu().item())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return per-parameter gradient norms after ``backward``.

    Parameters without a gradient are reported as zero. This makes the
    output stable for a layer that is inactive on a particular probe batch.
    """
    result: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            result[name] = 0.0
        else:
            result[name] = float(
                parameter.grad.detach().norm().cpu().item()
            )
    return result


def capture_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters to CPU for a later update measurement."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_norms(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure ``||parameter_after - parameter_before||`` per tensor."""
    result: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in before:
            raise KeyError(
                f"Missing parameter {name!r} in the saved state."
            )
        difference = parameter.detach().cpu() - before[name]
        result[name] = float(difference.norm().item())
    return result
