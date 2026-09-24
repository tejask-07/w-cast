"""Open and normalize GRIB2 datasets without assuming weather variable names."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	import xarray as xr


def inspect_dataset(dataset: xr.Dataset) -> dict[str, Any]:
	"""Describe variables, coordinates, and dimensions before mapping fields."""
	return {
		"data_vars": list(dataset.data_vars),
		"coords": list(dataset.coords),
		"dims": dict(dataset.sizes),
		"attrs": dict(dataset.attrs),
	}


def open_grib(path: str | Path, **kwargs: Any) -> xr.Dataset:
	"""Open a GRIB2 file with cfgrib and return a normalized Dataset."""
	try:
		import xarray as xr

		dataset = xr.open_dataset(path, engine="cfgrib", **kwargs)
	except Exception as exc:
		raise ValueError(f"Could not open GRIB2 file '{path}': {exc}") from exc
	return normalize_dataset(dataset)


def normalize_dataset(dataset: xr.Dataset) -> xr.Dataset:
	"""Normalize common latitude/longitude and time coordinate names.

	GFS/cfgrib commonly exposes ``latitude``, ``longitude``, ``time``,
	``step``, and ``valid_time``. Variable names are intentionally preserved;
	callers should use :func:`inspect_dataset` and choose a field explicitly.
	"""
	rename: dict[str, str] = {}
	if "latitude" in dataset and "lat" not in dataset:
		rename["latitude"] = "lat"
	if "longitude" in dataset and "lon" not in dataset:
		rename["longitude"] = "lon"
	if rename:
		dataset = dataset.rename(rename)
	if "valid_time" in dataset and "valid_time" not in dataset.coords:
		dataset = dataset.set_coords("valid_time")
	if "forecast_time" not in dataset.coords:
		if "time" in dataset.coords:
			dataset = dataset.assign_coords(forecast_time=dataset["time"])
		elif "valid_time" in dataset.coords:
			dataset = dataset.assign_coords(forecast_time=dataset["valid_time"])
	return dataset


def close_dataset(dataset: xr.Dataset) -> None:
	"""Close an opened GRIB dataset when the caller no longer needs it."""
	dataset.close()


def extract_nearest_grid_point(
	dataset: xr.Dataset, latitude: float, longitude: float
) -> xr.Dataset:
	"""Select the nearest normalized GFS grid point for any coordinates."""
	if "lat" not in dataset.coords or "lon" not in dataset.coords:
		raise ValueError("Dataset must contain normalized lat and lon coordinates")
	return dataset.sel(lat=latitude, lon=longitude, method="nearest")


LOCATION_METADATA = {
	"Mumbai": {"latitude": 19.0760, "longitude": 72.8777, "region": "west_coast"},
	"Pune": {"latitude": 18.5204, "longitude": 73.8567, "region": "west"},
	"Ahmedabad": {"latitude": 23.0225, "longitude": 72.5714, "region": "west"},
	"Jaipur": {"latitude": 26.9124, "longitude": 75.7873, "region": "northwest"},
	"Delhi": {"latitude": 28.6139, "longitude": 77.2090, "region": "north"},
	"Chandigarh": {"latitude": 30.7333, "longitude": 76.7794, "region": "north"},
	"Lucknow": {"latitude": 26.8467, "longitude": 80.9462, "region": "north"},
	"Srinagar": {"latitude": 34.0837, "longitude": 74.7973, "region": "himalayan"},
	"Dehradun": {"latitude": 30.3165, "longitude": 78.0322, "region": "himalayan"},
	"Bhopal": {"latitude": 23.2599, "longitude": 77.4126, "region": "central"},
	"Nagpur": {"latitude": 21.1458, "longitude": 79.0882, "region": "central"},
	"Kolkata": {"latitude": 22.5726, "longitude": 88.3639, "region": "east"},
	"Bhubaneswar": {"latitude": 20.2961, "longitude": 85.8245, "region": "east_coast"},
	"Guwahati": {"latitude": 26.1445, "longitude": 91.7362, "region": "northeast"},
	"Patna": {"latitude": 25.5941, "longitude": 85.1376, "region": "east"},
	"Chennai": {"latitude": 13.0827, "longitude": 80.2707, "region": "southeast_coast"},
	"Bengaluru": {"latitude": 12.9716, "longitude": 77.5946, "region": "south"},
	"Hyderabad": {"latitude": 17.3850, "longitude": 78.4867, "region": "south_central"},
	"Kochi": {"latitude": 9.9312, "longitude": 76.2673, "region": "southwest_coast"},
	"Visakhapatnam": {"latitude": 17.6868, "longitude": 83.2185, "region": "east_coast"},
}

INDIA_LOCATIONS = tuple(LOCATION_METADATA)

CITIES = {
	city: (
		LOCATION_METADATA[city]["latitude"],
		LOCATION_METADATA[city]["longitude"],
	)
	for city in ("Mumbai", "Delhi", "Kolkata", "Chennai")
}


def extract_city_grid_point(dataset: xr.Dataset, city: str) -> xr.Dataset:
	"""Select a named MVP city using the nearest grid point."""
	try:
		metadata = LOCATION_METADATA[city]
	except KeyError as exc:
		raise ValueError(f"Unknown city '{city}'. Available cities: {sorted(LOCATION_METADATA)}") from exc
	return extract_nearest_grid_point(
		dataset,
		metadata["latitude"],
		metadata["longitude"],
	)
