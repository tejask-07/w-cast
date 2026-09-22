"""Weather-regime feature helpers."""


def regime_features(regime: str) -> dict[str, str]:
	"""Return a validated categorical regime feature."""
	normalized = regime.upper()
	if normalized not in {"DRY", "NORMAL", "WET", "EXTREME"}:
		raise ValueError(f"Unknown regime: {regime}")
	return {"regime": normalized}
