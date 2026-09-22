"""Equal-weight forecast baseline."""

from typing import Any, Mapping

import numpy as np


def equal_weights(model_names: list[str] | tuple[str, ...]) -> dict[str, float]:
	"""Return uniform weights for the supplied model names."""
	if not model_names:
		raise ValueError("At least one model is required")
	return {name: 1.0 / len(model_names) for name in model_names}


def equal_weight_blend(forecasts: Mapping[str, Any]) -> np.ndarray:
	"""Average compatible forecast values across models."""
	return _blend(forecasts, equal_weights(list(forecasts)))


def _blend(forecasts: Mapping[str, Any], weights: Mapping[str, float]) -> np.ndarray:
	if not forecasts:
		raise ValueError("At least one forecast is required")
	values = [np.asarray(forecasts[name], dtype=float) for name in weights]
	return np.mean(np.stack(values, axis=0), axis=0)
