"""Forward-hook activation statistics for toy-sized probe batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActivationStats:
    """Online summary of one module's tensor output."""

    module_name: str
    numel: int
    mean: float
    std: float
    rms: float
    zero_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Return a flat row suitable for CSV output."""
        return {
            "module_name": self.module_name,
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "rms": self.rms,
            "zero_fraction": self.zero_fraction,
        }


class ActivationRecorder:
    """Record selected module outputs without retaining autograd graphs.

    Hooks only collect scalar sums by default. ``store_samples=True`` is
    intended for a small fixed probe set when a later PCA plot needs the
    actual hidden vectors.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        module_types: tuple[type[nn.Module], ...] = (nn.ReLU,),
        store_samples: bool = False,
    ) -> None:
        self._store_samples = store_samples
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._squared_sums: dict[str, float] = {}
        self._zero_counts: dict[str, int] = {}
        self._samples: dict[str, list[torch.Tensor]] = {}

        for module_name, module in model.named_modules():
            if isinstance(module, module_types):
                handle = module.register_forward_hook(
                    self._make_hook(module_name)
                )
                self._handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from {module_name!r}."
                )
            values = output.detach().float()
            flattened = values.reshape(-1)
            count = int(flattened.numel())
            self._counts[module_name] = (
                self._counts.get(module_name, 0) + count
            )
            self._sums[module_name] = (
                self._sums.get(module_name, 0.0)
                + float(flattened.sum().cpu().item())
            )
            self._squared_sums[module_name] = (
                self._squared_sums.get(module_name, 0.0)
                + float(flattened.square().sum().cpu().item())
            )
            self._zero_counts[module_name] = (
                self._zero_counts.get(module_name, 0)
                + int((flattened == 0).sum().cpu().item())
            )
            if self._store_samples:
                self._samples.setdefault(module_name, []).append(
                    values.cpu().clone()
                )

        return hook

    def remove(self) -> None:
        """Remove every hook; safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.remove()

    def stats(self) -> list[ActivationStats]:
        """Return summaries in module traversal order."""
        result: list[ActivationStats] = []
        for module_name in self._counts:
            count = self._counts[module_name]
            mean = self._sums[module_name] / count
            second_moment = self._squared_sums[module_name] / count
            variance = max(second_moment - mean * mean, 0.0)
            result.append(
                ActivationStats(
                    module_name=module_name,
                    numel=count,
                    mean=mean,
                    std=variance**0.5,
                    rms=second_moment**0.5,
                    zero_fraction=self._zero_counts[module_name] / count,
                )
            )
        return result

    def samples(self) -> dict[str, torch.Tensor]:
        """Return concatenated detached samples collected by the hooks."""
        if not self._store_samples:
            raise RuntimeError(
                "Construct the recorder with store_samples=True first."
            )
        return {
            module_name: torch.cat(values, dim=0)
            for module_name, values in self._samples.items()
        }


def record_activation_stats(
    model: nn.Module,
    x: torch.Tensor,
    *,
    module_types: tuple[type[nn.Module], ...] = (nn.ReLU,),
    store_samples: bool = False,
) -> ActivationRecorder:
    """Run one probe batch and return a recorder with hooks removed."""
    model.eval()
    recorder = ActivationRecorder(
        model,
        module_types=module_types,
        store_samples=store_samples,
    )
    try:
        with torch.no_grad():
            model(x)
    finally:
        recorder.remove()
    return recorder
