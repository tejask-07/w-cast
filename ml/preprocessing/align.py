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


CITIES = {
	"Mumbai": (19.0760, 72.8777),
	"Delhi": (28.6139, 77.2090),
	"Kolkata": (22.5726, 88.3639),
	"Chennai": (13.0827, 80.2707),
}


def extract_city_grid_point(dataset: xr.Dataset, city: str) -> xr.Dataset:
	"""Select a named MVP city using the nearest grid point."""
	try:
		latitude, longitude = CITIES[city]
	except KeyError as exc:
		raise ValueError(f"Unknown city '{city}'. Available cities: {sorted(CITIES)}") from exc
	return extract_nearest_grid_point(dataset, latitude, longitude)
