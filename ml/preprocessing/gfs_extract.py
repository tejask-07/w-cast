"""Memory-efficient extraction from verified GFS GRIB subset files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FieldPaths = Mapping[str, str | Path]


def open_gfs_subset(path: str | Path):
    """Open one downloaded GFS subset with cfgrib."""
    import xarray as xr

    return xr.open_dataset(path, engine="cfgrib")


def _coordinate_name(dataset: Any, long_name: str, short_name: str) -> str:
    available = set(getattr(dataset, "coords", {}))
    available.update(getattr(dataset, "variables", {}))
    available.update(getattr(dataset, "dims", {}))
    for name in (long_name, short_name):
        if name in available:
            return name
    raise ValueError(f"Missing {long_name}/{short_name} coordinate; available={sorted(available)}")


def normalize_longitude(dataset: Any, longitude: float) -> float:
    """Normalize a requested longitude to GFS's 0-360 convention."""
    value = float(longitude)
    if not -180 <= value <= 360:
        raise ValueError("longitude must be between -180 and 360 degrees")
    longitude_name = _coordinate_name(dataset, "longitude", "lon")
    coordinate = dataset[longitude_name]
    coordinates = np.asarray(getattr(coordinate, "values", coordinate))
    if coordinates.size == 0:
        raise ValueError("GFS subset has no longitude coordinates")
    if float(np.nanmin(coordinates)) >= 0:
        return value % 360
    return value


def _gfs_longitude(dataset: Any, longitude: float) -> float:
    """Backward-compatible alias for longitude normalization."""
    return normalize_longitude(dataset, longitude)


def _read_subset_value(
    path: str | Path | None,
    latitude: float,
    longitude: float,
    variable_names: tuple[str, ...],
) -> float | None:
    """Read one nearest scalar from a subset, returning None if unavailable."""
    if path is None:
        return None
    dataset = None
    try:
        dataset = open_gfs_subset(path)
        variable_name = next((name for name in variable_names if name in dataset.data_vars), None)
        if variable_name is None:
            return None
        latitude_name = _coordinate_name(dataset, "latitude", "lat")
        longitude_name = _coordinate_name(dataset, "longitude", "lon")
        selected = dataset.sel(
            {
                latitude_name: float(latitude),
                longitude_name: normalize_longitude(dataset, longitude),
            },
            method="nearest",
        )
        value = np.asarray(selected[variable_name].values).squeeze()
        if value.size != 1:
            raise ValueError(f"Nearest selection for {variable_name} was not scalar: {value.shape}")
        return float(value.item())
    except (FileNotFoundError, OSError, ValueError):
        return None
    finally:
        if dataset is not None:
            dataset.close()


def _field_path(file_path: str | Path | FieldPaths, name: str) -> str | Path | None:
    if isinstance(file_path, Mapping):
        return file_path.get(name)
    return file_path


def extract_gfs_point(
    file_path: str | Path | FieldPaths,
    lat: float,
    lon: float,
    lead_hours: int | None = None,
) -> dict[str, Any]:
    """Extract t2m, u10, v10, and accumulated tp at the nearest grid point."""
    if isinstance(file_path, Mapping):
        if not file_path:
            raise ValueError("At least one GFS subset path is required")
    elif not Path(file_path).is_file():
        raise FileNotFoundError(f"GFS GRIB2 file does not exist: {file_path}")
    if not -90 <= float(lat) <= 90:
        raise ValueError("lat must be between -90 and 90 degrees")
    if lead_hours is not None and lead_hours < 0:
        raise ValueError("lead_hours must be non-negative")

    temperature_k = _read_subset_value(_field_path(file_path, "temperature"), lat, lon, ("t2m", "2t"))
    wind_u = _read_subset_value(_field_path(file_path, "wind_u"), lat, lon, ("u10",))
    wind_v = _read_subset_value(_field_path(file_path, "wind_v"), lat, lon, ("v10",))
    precipitation_kg_m2 = _read_subset_value(_field_path(file_path, "precipitation"), lat, lon, ("tp",))
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "lead_hours": lead_hours,
        "temperature_C": None if temperature_k is None else temperature_k - 273.15,
        "temperature_source": "GFS TMP 2m",
        "wind_u_ms": wind_u,
        "wind_v_ms": wind_v,
        "wind_speed_ms": None if wind_u is None or wind_v is None else float(np.hypot(wind_u, wind_v)),
        "wind_source": "GFS UGRD/VGRD 10m",
        "precipitation_mm": precipitation_kg_m2,
        "precipitation_source": "GFS APCP 0-1 day",
        "field_status": {
            "temperature": "available" if temperature_k is not None else "unavailable",
            "wind": "available" if wind_u is not None and wind_v is not None else "unavailable",
            "precipitation": "available" if precipitation_kg_m2 is not None else "unavailable",
        },
    }
def extract_gfs_grid(
    file_path: str | Path | FieldPaths,
    bounds: tuple[float, float, float, float] = (
        8.0,
        37.5,
        68.0,
        97.5,
    ),
) -> dict[str, Any]:
    """
    Extract native-resolution GFS grids inside geographic bounds.

    bounds:
        (latitude_min, latitude_max, longitude_min, longitude_max)
    """

    latitude_min = float(bounds[0])
    latitude_max = float(bounds[1])
    longitude_min = float(bounds[2])
    longitude_max = float(bounds[3])

    if latitude_min >= latitude_max:
        raise ValueError(
            "latitude_min must be less than latitude_max"
        )

    if longitude_min >= longitude_max:
        raise ValueError(
            "longitude_min must be less than longitude_max"
        )

    fields = {
        "temperature": ("t2m", "2t"),
        "wind_u": ("u10",),
        "wind_v": ("v10",),
        "precipitation": ("tp",),
    }

    result: dict[str, Any] = {}

    for field, variable_names in fields.items():
        path = _field_path(file_path, field)

        if path is None:
            result[field] = None
            continue

        if not Path(path).is_file():
            result[field] = None
            continue

        dataset = None

        try:
            dataset = open_gfs_subset(path)

            variable_name = next(
                (
                    name
                    for name in variable_names
                    if name in dataset.data_vars
                ),
                None,
            )

            if variable_name is None:
                result[field] = None
                continue

            latitude_name = _coordinate_name(
                dataset,
                "latitude",
                "lat",
            )

            longitude_name = _coordinate_name(
                dataset,
                "longitude",
                "lon",
            )

            latitude = dataset[latitude_name]
            longitude = dataset[longitude_name]

            longitude_min_normalized = normalize_longitude(
                dataset,
                longitude_min,
            )

            longitude_max_normalized = normalize_longitude(
                dataset,
                longitude_max,
            )

            latitude_mask = (
                (latitude >= latitude_min)
                & (latitude <= latitude_max)
            )

            longitude_mask = (
                (longitude >= longitude_min_normalized)
                & (longitude <= longitude_max_normalized)
            )

            selected = dataset[variable_name].where(
                latitude_mask & longitude_mask,
                drop=True,
            ).load()

            result[field] = selected

        finally:
            if dataset is not None:
                dataset.close()

    return result
