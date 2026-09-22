"""Historical-error adaptive baseline, not a trained AI model."""

from typing import Mapping

import numpy as np


def adaptive_weights(
	errors: Mapping[str, float | None], epsilon: float = 1e-6
) -> dict[str, float]:
	"""Convert historical errors into inverse-error normalized weights.

	Missing or non-finite errors are excluded. If every error is missing, the
	method falls back to equal weights so it remains usable before history exists.
	"""
	if epsilon <= 0:
		raise ValueError("epsilon must be positive")
	if not errors:
		raise ValueError("At least one model error is required")
	valid = {
		name: float(error)
		for name, error in errors.items()
		if error is not None and np.isfinite(error) and error >= 0
	}
	if not valid:
		uniform = 1.0 / len(errors)
		return {name: uniform for name in errors}
	scores = {name: 1.0 / (error + epsilon) for name, error in valid.items()}
	total = sum(scores.values())
	weights = {name: score / total for name, score in scores.items()}
	missing = set(errors) - set(weights)
	if missing:
		for name in missing:
			weights[name] = 0.0
	return weights
