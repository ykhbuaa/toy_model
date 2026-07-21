"""Model evaluation utilities."""

from toy_model.evaluation.classification import (
    ClassificationMetrics,
    compute_confusion_matrix,
    compute_logit_margins,
    evaluate_classification,
    evaluate_classification_logits,
)
from toy_model.evaluation.regression import (
    FunctionRegressionMetrics,
    evaluate_function_regression,
)

from toy_model.evaluation.extrapolation import (
    AffineFitMetrics,
    BoundaryDistanceMetric,
    SymmetryMetrics,
    compute_boundary_distance_metrics,
    compute_fixed_zone_metrics,
    compute_periodicity_residual,
    compute_support_metrics,
    compute_symmetry_metrics,
    compute_tail_affine_metrics,
    fit_affine_function,
)

__all__ = [
    "FunctionRegressionMetrics",
    "evaluate_function_regression",
    "AffineFitMetrics",
    "BoundaryDistanceMetric",
    "SymmetryMetrics",
    "compute_boundary_distance_metrics",
    "compute_fixed_zone_metrics",
    "compute_periodicity_residual",
    "compute_support_metrics",
    "compute_symmetry_metrics",
    "compute_tail_affine_metrics",
    "fit_affine_function",
    "ClassificationMetrics",
    "compute_confusion_matrix",
    "compute_logit_margins",
    "evaluate_classification",
    "evaluate_classification_logits",
]
