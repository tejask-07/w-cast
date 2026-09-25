from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from app.services.forecast_service import (
    REPO_ROOT,
    _download_forecast_sources,
    resolve_location_name,
)
from ml.blending.blender import blend_forecasts
from ml.evaluation.historical_weights import build_historical_weights
from ml.spatial.india_grid import is_in_india

SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")
SUPPORTED_LEADS = (24, 48, 72)
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"
CACHE_DIR = REPO_ROOT / "data" / "interim" / "forecast_maps"
REGION_HALF_SIZE_DEGREES = 5.0
SOURCE_REGION_MARGIN_DEGREES = 1.0


class SpatialForecastError(RuntimeError):
    """Raised when a real spatial forecast cannot be generated."""


def validate_spatial_request(
    variable: str,
    lead_hours: int,
    south: float | None = None,
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")
    if lead_hours not in SUPPORTED_LEADS:
        raise ValueError("lead_hours must be one of 24, 48, or 72")
    _validate_bounds(south, north, west, east)


def _coordinate_name(dataset, long_name: str, short_name: str) -> str:
    available = set(dataset.coords)
    available.update(dataset.variables)
    available.update(dataset.dims)
    for name in (long_name, short_name):
        if name in available:
            return name
    raise SpatialForecastError(
        f"Missing {long_name}/{short_name} coordinate in GRIB field"
    )


def _read_data_array(path: str | Path, names: tuple[str, ...]):
    import xarray as xr

    try:
        dataset = xr.open_dataset(path, engine="cfgrib")
    except Exception as exc:
        raise SpatialForecastError(f"Unable to open forecast field {path}") from exc

    try:
        variable_name = next(
            (name for name in names if name in dataset.data_vars),
            None,
        )
        if variable_name is None:
            raise SpatialForecastError(
                f"Forecast field {names[0]} is unavailable in {path}"
            )

        latitude_name = _coordinate_name(dataset, "latitude", "lat")
        longitude_name = _coordinate_name(dataset, "longitude", "lon")
        data = dataset[variable_name].squeeze(drop=True)
        if data.ndim != 2:
            raise SpatialForecastError(
                f"Forecast field {variable_name} is not a 2D grid"
            )

        data = data.transpose(latitude_name, longitude_name).load()
        latitudes = np.asarray(data[latitude_name].values, dtype=float)
        longitudes = np.asarray(data[longitude_name].values, dtype=float)
        values = np.asarray(data.values, dtype=float)
    finally:
        dataset.close()

    if latitudes.size < 2 or longitudes.size < 2:
        raise SpatialForecastError("Forecast field does not contain a usable grid")

    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        values = values[::-1, :]

    longitudes = ((longitudes + 180.0) % 360.0) - 180.0
    longitude_order = np.argsort(longitudes)
    longitudes = longitudes[longitude_order]
    values = values[:, longitude_order]

    return latitudes, longitudes, values


def _read_model_field(paths: Mapping[str, str | Path], variable: str):
    if variable == "temperature":
        latitudes, longitudes, values = _read_data_array(
            paths["temperature"],
            ("t2m", "2t"),
        )
        return latitudes, longitudes, values - 273.15

    if variable == "rainfall":
        return _read_data_array(paths["precipitation"], ("tp",))

    latitudes, longitudes, wind_u = _read_data_array(paths["wind_u"], ("u10",))
    other_latitudes, other_longitudes, wind_v = _read_data_array(
        paths["wind_v"],
        ("v10",),
    )
    if not (
        np.array_equal(latitudes, other_latitudes)
        and np.array_equal(longitudes, other_longitudes)
    ):
        raise SpatialForecastError("GFS wind fields do not share a grid")
    return latitudes, longitudes, np.hypot(wind_u, wind_v)


def _crop_region(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    latitude: float,
    longitude: float,
    bounds: tuple[float, float, float, float] | None = None,
):
    if bounds is None:
        south = latitude - REGION_HALF_SIZE_DEGREES
        north = latitude + REGION_HALF_SIZE_DEGREES
        west = longitude - REGION_HALF_SIZE_DEGREES
        east = longitude + REGION_HALF_SIZE_DEGREES
    else:
        south, north, west, east = bounds
        south -= SOURCE_REGION_MARGIN_DEGREES
        north += SOURCE_REGION_MARGIN_DEGREES
        west -= SOURCE_REGION_MARGIN_DEGREES
        east += SOURCE_REGION_MARGIN_DEGREES

    latitude_mask = (
        (latitudes >= south)
        & (latitudes <= north)
    )
    longitude_mask = (
        (longitudes >= west)
        & (longitudes <= east)
    )
    if latitude_mask.sum() < 2 or longitude_mask.sum() < 2:
        raise SpatialForecastError(
            "Requested location is outside the available forecast grid"
        )
    return (
        latitudes[latitude_mask],
        longitudes[longitude_mask],
        values[np.ix_(latitude_mask, longitude_mask)],
    )


def _validate_bounds(
    south: float | None,
    north: float | None,
    west: float | None,
    east: float | None,
) -> tuple[float, float, float, float] | None:
    values = (south, north, west, east)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("south, north, west, and east must be supplied together")
    assert south is not None and north is not None
    assert west is not None and east is not None
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("AOI latitude bounds must be between -90 and 90")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("AOI longitude bounds must be between -180 and 180")
    if south >= north:
        raise ValueError("AOI south must be less than north")
    if west >= east:
        raise ValueError("AOI west must be less than east")
    return south, north, west, east


def _interpolate_to_grid(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
) -> np.ndarray:
    import xarray as xr

    source = xr.DataArray(
        values,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    )
    return np.asarray(
        source.interp(
            latitude=target_latitudes,
            longitude=target_longitudes,
            method="linear",
        ).values,
        dtype=float,
    )


def _weights(latitude: float, longitude: float, variable: str, lead_hours: int):
    if not is_in_india(latitude, longitude):
        return {"gfs": 0.5, "gefs": 0.5}
    city = resolve_location_name(latitude, longitude)
    internal_variable = "precipitation" if variable == "rainfall" else variable
    historical = build_historical_weights(HISTORY_PATH)
    return historical.get(city, {}).get(internal_variable, {}).get(
        str(lead_hours), {}
    ).get("weights", {"gfs": 0.5, "gefs": 0.5})


def _cache_key(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    gfs_paths: Mapping[str, str | Path],
    gefs_paths: Mapping[str, str | Path],
    bounds: tuple[float, float, float, float] | None,
) -> str:
    source_signature = []
    for model, paths in (("gfs", gfs_paths), ("gefs", gefs_paths)):
        for name, path in sorted(paths.items()):
            source = Path(path)
            stat = source.stat()
            source_signature.append(
                (model, name, source.name, stat.st_size, stat.st_mtime_ns)
            )
    payload = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "variable": variable,
        "lead_hours": lead_hours,
        "bounds": bounds,
        "sources": source_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def _render_png(
    output_path: Path,
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    variable: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colormaps = {
        "temperature": "RdYlBu_r",
        "rainfall": "Blues",
        "wind_speed": "viridis",
    }
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise SpatialForecastError("Forecast field contains no finite values")

    figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
    try:
        axis.imshow(
            values,
            origin="lower",
            extent=(
                float(longitudes.min()),
                float(longitudes.max()),
                float(latitudes.min()),
                float(latitudes.max()),
            ),
            cmap=colormaps[variable],
            interpolation="bilinear",
            aspect="auto",
        )
        axis.axis("off")
        figure.savefig(
            output_path,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    finally:
        plt.close(figure)


def generate_spatial_forecast_map(
    latitude: float,
    longitude: float,
    lead_hours: int,
    variable: str,
    south: float | None = None,
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
) -> dict[str, object]:
    validate_spatial_request(
        variable,
        lead_hours,
        south,
        north,
        west,
        east,
    )
    bounds = _validate_bounds(south, north, west, east)
    gfs_paths, gefs_paths = _download_forecast_sources(lead_hours)
    cache_key = _cache_key(
        latitude,
        longitude,
        variable,
        lead_hours,
        gfs_paths,
        gefs_paths,
        bounds,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = CACHE_DIR / f"{cache_key}.png"
    metadata_path = CACHE_DIR / f"{cache_key}.json"

    if image_path.is_file() and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            cached = json.load(file)
        if {"min_value", "max_value", "unit"}.issubset(cached):
            return cached

    gfs_latitudes, gfs_longitudes, gfs_values = _read_model_field(
        gfs_paths,
        variable,
    )
    gefs_latitudes, gefs_longitudes, gefs_values = _read_model_field(
        gefs_paths,
        variable,
    )
    latitudes, longitudes, gfs_values = _crop_region(
        gfs_latitudes,
        gfs_longitudes,
        gfs_values,
        latitude,
        longitude,
        bounds,
    )
    if bounds is None:
        target_latitudes = latitudes
        target_longitudes = longitudes
        display_bounds = (
            float(latitudes.min()),
            float(latitudes.max()),
            float(longitudes.min()),
            float(longitudes.max()),
        )
    else:
        target_latitudes = np.linspace(bounds[0], bounds[1], max(2, round((bounds[1] - bounds[0]) / 0.25) + 1))
        target_longitudes = np.linspace(bounds[2], bounds[3], max(2, round((bounds[3] - bounds[2]) / 0.25) + 1))
        gfs_values = _interpolate_to_grid(
            latitudes,
            longitudes,
            gfs_values,
            target_latitudes,
            target_longitudes,
        )
        display_bounds = bounds

    gefs_values = _interpolate_to_grid(
        gefs_latitudes,
        gefs_longitudes,
        gefs_values,
        target_latitudes,
        target_longitudes,
    )
    weights = _weights(latitude, longitude, variable, lead_hours)
    blended_values = blend_forecasts(
        {"gfs": gfs_values, "gefs": gefs_values},
        weights,
    )
    finite_values = blended_values[np.isfinite(blended_values)]
    if finite_values.size == 0:
        raise SpatialForecastError("Forecast field contains no finite values")
    units = {
        "temperature": "°C",
        "rainfall": "mm",
        "wind_speed": "m/s",
    }
    _render_png(
        image_path,
        blended_values,
        target_latitudes,
        target_longitudes,
        variable,
    )

    result = {
        "variable": variable,
        "lead_hours": lead_hours,
        "bounds": [
            [display_bounds[0], display_bounds[2]],
            [display_bounds[1], display_bounds[3]],
        ],
        "image_url": f"/api/forecast/map/image/{cache_key}.png",
        "min_value": float(np.min(finite_values)),
        "max_value": float(np.max(finite_values)),
        "unit": units[variable],
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(result, file)
    return result


def get_cached_map_path(cache_name: str) -> Path:
    if not cache_name.endswith(".png") or Path(cache_name).name != cache_name:
        raise FileNotFoundError(cache_name)
    path = CACHE_DIR / cache_name
    if not path.is_file():
        raise FileNotFoundError(cache_name)
    return path
