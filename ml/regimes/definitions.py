"""Configurable precipitation regime thresholds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeThresholds:
	"""Rainfall thresholds in millimetres for the classification window."""

	dry_max_mm: float = 2.5
	normal_max_mm: float = 15.6
	wet_max_mm: float = 64.5


DEFAULT_THRESHOLDS = RegimeThresholds()
REGIMES = ("DRY", "NORMAL", "WET", "EXTREME")
