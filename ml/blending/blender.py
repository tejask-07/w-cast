"""Shared forecast blending primitive."""

from typing import Any, Mapping

import numpy as np


def validate_weights(
	forecasts: Mapping[str, Any], weights: Mapping[str, float], tolerance: float = 1e-6
) -> None:
	"""Validate non-negative, complete, approximately normalized weights."""
	if not forecasts:
		raise ValueError("At least one forecast is required")
	if set(forecasts) != set(weights):
		raise ValueError("Forecast and weight model names must match")
	numeric_weights = np.asarray(list(weights.values()), dtype=float)
	if not np.all(np.isfinite(numeric_weights)) or np.any(numeric_weights < 0):
		raise ValueError("All weights must be finite and non-negative")
	if not np.isclose(numeric_weights.sum(), 1.0, atol=tolerance):
		raise ValueError("Weights must sum to approximately 1")

	values = [np.asarray(forecasts[name]) for name in forecasts]
	if any(value.shape != values[0].shape for value in values[1:]):
		raise ValueError("All forecasts must have compatible shapes")
	if any(not np.issubdtype(value.dtype, np.number) for value in values):
		raise ValueError("All forecasts must contain numeric values")


def blend_forecasts(
	forecasts: Mapping[str, Any], weights: Mapping[str, float]
) -> np.ndarray:
	"""Return ``sum(weight_i * forecast_i)`` after validating inputs."""
	validate_weights(forecasts, weights)
	result = np.zeros_like(np.asarray(next(iter(forecasts.values())), dtype=float))
	for name, forecast in forecasts.items():
		result = result + weights[name] * np.asarray(forecast, dtype=float)
	return result
