"""Calendar and lead-time features."""

from datetime import datetime
from typing import Any

import pandas as pd


SEASONS = ("winter", "pre_monsoon", "monsoon", "post_monsoon")


def get_season(value: datetime) -> str:
	"""Return India's climatological season for a timestamp month."""
	month = value.month
	if month in (12, 1, 2):
		return "winter"
	if month in (3, 4, 5):
		return "pre_monsoon"
	if month in (6, 7, 8, 9):
		return "monsoon"
	return "post_monsoon"


def temporal_features(timestamp: Any, lead_hours: int) -> dict[str, Any]:
	"""Return timestamp, month, season, and lead-time features."""
	time = pd.Timestamp(timestamp)
	if lead_hours < 0:
		raise ValueError("lead_hours must be non-negative")
	season = get_season(time.to_pydatetime())
	return {"forecast_time": time, "lead_hours": lead_hours, "month": time.month, "season": season}
