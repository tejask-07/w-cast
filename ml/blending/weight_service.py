"""Canonical historical-weight lookup shared by forecast consumers."""

from pathlib import Path
from typing import Any

from ml.spatial.lookup import get_spatial_weights


def get_forecast_weights(
	latitude: float,
	longitude: float,
	variable: str,
	lead_hours: int,
	season: str | None = None,
	regime: str | None = None,
	path: str | Path | None = None,
) -> dict[str, Any]:
	"""Return weights and provenance from the canonical spatial artifact.

	Season and regime are accepted as context dimensions so callers share one
	interface; the current artifact has lead/variable granularity only.
	"""
	kwargs = {} if path is None else {"path": path}
	result = get_spatial_weights(
		latitude=latitude,
		longitude=longitude,
		variable=variable,
		lead_hours=lead_hours,
		**kwargs,
	)
	result["season"] = season
	result["regime"] = regime
	return result