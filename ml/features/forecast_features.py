"""Feature row assembly for a future learned adaptive-weight model."""

from typing import Any, Mapping

from ml.features.regime_features import regime_features
from ml.features.spatial_features import spatial_features
from ml.features.temporal_features import temporal_features


def forecast_features(
	source: str,
	latitude: float,
	longitude: float,
	lead_hours: int,
	forecast_time: Any,
	regime: str,
	forecast_value: float,
	recent_model_error: float | None = None,
	ensemble_spread: float | None = None,
) -> dict[str, Any]:
	"""Build one explicit feature row; unavailable inputs remain ``None``."""
	features: dict[str, Any] = {
		"source": source,
		"forecast_value": float(forecast_value),
		"recent_model_error": recent_model_error,
		"ensemble_spread": ensemble_spread,
	}
	features.update(spatial_features(latitude, longitude))
	features.update(temporal_features(forecast_time, lead_hours))
	features.update(regime_features(regime))
	return features


def build_feature_frame(rows: list[Mapping[str, Any]]):
	"""Convert feature rows to a pandas DataFrame without inventing missing data."""
	import pandas as pd

	return pd.DataFrame(rows)
