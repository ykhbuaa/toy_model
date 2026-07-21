"""Training utilities."""

from toy_model.training.classification import (
    ClassificationTrainConfig,
    ClassificationTrainResult,
    fit_classification,
    predict_labels,
    predict_logits,
)
from toy_model.training.regression import (
    RegressionTrainConfig,
    RegressionTrainResult,
    evaluate_mse,
    fit_regression,
    predict,
    resolve_device,
)

__all__ = [
    "RegressionTrainConfig",
    "RegressionTrainResult",
    "evaluate_mse",
    "fit_regression",
    "predict",
    "resolve_device",
    "ClassificationTrainConfig",
    "ClassificationTrainResult",
    "fit_classification",
    "predict_labels",
    "predict_logits",
]
