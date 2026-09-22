"""Validation for downloaded GFS GRIB2 files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
	import xarray as xr

from ml.preprocessing.align import inspect_dataset, open_grib


def validate_grib(
	path: str | Path,
	required_variables: Iterable[str] = (),
) -> xr.Dataset:
	"""Validate and return a normalized GRIB dataset.

	The returned dataset remains open; callers should close it after use.
	"""
	file_path = Path(path)
	if not file_path.is_file():
		raise FileNotFoundError(f"GRIB2 file does not exist: {file_path}")
	dataset = open_grib(file_path)
	details = inspect_dataset(dataset)
	missing_coords = {"lat", "lon"} - set(details["coords"])
	if missing_coords:
		dataset.close()
		raise ValueError(f"GRIB2 is missing latitude/longitude coordinates: {sorted(missing_coords)}")
	if "forecast_time" not in details["coords"] and "valid_time" not in details["coords"]:
		dataset.close()
		raise ValueError("GRIB2 is missing forecast time (time or valid_time)")
	missing_variables = set(required_variables) - set(details["data_vars"])
	if missing_variables:
		dataset.close()
		raise ValueError(f"GRIB2 is missing required variables: {sorted(missing_variables)}")
	return dataset
