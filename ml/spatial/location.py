"""Canonical configured-location resolution for the India domain."""

from typing import Any

from ml.preprocessing.align import LOCATION_METADATA
from ml.spatial.india_grid import is_in_india


def resolve_location(latitude: float, longitude: float) -> dict[str, Any]:
	"""Return the nearest configured location, or reject coordinates outside India."""
	if not is_in_india(latitude, longitude):
		raise ValueError("Coordinates are outside the supported India domain")
	city = min(
		LOCATION_METADATA,
		key=lambda name: (
			(LOCATION_METADATA[name]["latitude"] - latitude) ** 2
			+ (LOCATION_METADATA[name]["longitude"] - longitude) ** 2
		),
	)
	metadata = LOCATION_METADATA[city]
	return {
		"city": city,
		"region": metadata["region"],
		"latitude": metadata["latitude"],
		"longitude": metadata["longitude"],
	}