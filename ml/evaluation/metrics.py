"""Forecast verification metrics."""

from typing import Any

import numpy as np


def _paired_values(actual: Any, predicted: Any) -> tuple[np.ndarray, np.ndarray]:
	observed = np.asarray(actual, dtype=float)
	forecast = np.asarray(predicted, dtype=float)
	if observed.shape != forecast.shape:
		raise ValueError("actual and predicted must have the same shape")
	mask = np.isfinite(observed) & np.isfinite(forecast)
	if not np.any(mask):
		raise ValueError("No finite actual/predicted pairs are available")
	return observed[mask], forecast[mask]


def mae(actual: Any, predicted: Any) -> float:
	"""Return mean absolute error, ignoring paired missing values."""
	observed, forecast = _paired_values(actual, predicted)
	return float(np.mean(np.abs(forecast - observed)))


def rmse(actual: Any, predicted: Any) -> float:
	"""Return root mean squared error, ignoring paired missing values."""
	observed, forecast = _paired_values(actual, predicted)
	return float(np.sqrt(np.mean((forecast - observed) ** 2)))


def bias(actual: Any, predicted: Any) -> float:
	"""Return mean forecast minus observation bias."""
	observed, forecast = _paired_values(actual, predicted)
	return float(np.mean(forecast - observed))
