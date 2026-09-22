"""Simple rule-based weather regime classifier."""

from numbers import Real

from ml.regimes.definitions import DEFAULT_THRESHOLDS, RegimeThresholds


def classify_regime(
	precipitation_mm: Real, thresholds: RegimeThresholds = DEFAULT_THRESHOLDS
) -> str:
	"""Classify accumulated precipitation as DRY, NORMAL, WET, or EXTREME."""
	value = float(precipitation_mm)
	if value < 0:
		raise ValueError("precipitation_mm cannot be negative")
	if value <= thresholds.dry_max_mm:
		return "DRY"
	if value <= thresholds.normal_max_mm:
		return "NORMAL"
	if value <= thresholds.wet_max_mm:
		return "WET"
	return "EXTREME"
