"""Compare named forecasts against observations."""

from typing import Any, Mapping

import pandas as pd

from ml.evaluation.metrics import bias, mae, rmse


def compare_forecasts(
	actual: Any, forecasts: Mapping[str, Any]
) -> pd.DataFrame:
	"""Return MAE, RMSE, and bias for each available forecast source."""
	rows = [
		{"model": name, "mae": mae(actual, values), "rmse": rmse(actual, values), "bias": bias(actual, values)}
		for name, values in forecasts.items()
	]
	return pd.DataFrame(rows, columns=["model", "mae", "rmse", "bias"])
