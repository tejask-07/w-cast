"""User-configured static forecast blending baseline."""

from typing import Any, Mapping

import numpy as np

from ml.blending.blender import blend_forecasts


def static_weight_blend(
	forecasts: Mapping[str, Any], weights: Mapping[str, float]
) -> np.ndarray:
	"""Blend forecasts using validated user-specified weights."""
	return blend_forecasts(forecasts, weights)
