"""Spatial features for a point forecast."""


def spatial_features(latitude: float, longitude: float) -> dict[str, float]:
	"""Return validated latitude and longitude features."""
	if not -90 <= latitude <= 90:
		raise ValueError("latitude must be between -90 and 90")
	if not -180 <= longitude <= 180:
		raise ValueError("longitude must be between -180 and 180")
	return {"latitude": float(latitude), "longitude": float(longitude)}
