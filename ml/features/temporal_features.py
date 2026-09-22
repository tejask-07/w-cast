"""Calendar and lead-time features."""

from typing import Any

import pandas as pd


def temporal_features(timestamp: Any, lead_hours: int) -> dict[str, Any]:
	"""Return timestamp, month, season, and lead-time features."""
	time = pd.Timestamp(timestamp)
	if lead_hours < 0:
		raise ValueError("lead_hours must be non-negative")
	season = {
		12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
		6: "JJA", 7: "JJA", 8: "JJA",
	}.get(time.month, "SON")
	return {"forecast_time": time, "lead_hours": lead_hours, "month": time.month, "season": season}
