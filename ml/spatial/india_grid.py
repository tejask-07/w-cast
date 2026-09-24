"""Configurable India grid generation using a compact boundary mask."""

from __future__ import annotations

from typing import Iterable

from ml.preprocessing.align import LOCATION_METADATA

DEFAULT_BOUNDS = (8.0, 37.5, 68.0, 97.5)
DEFAULT_RESOLUTION = 0.5

# A compact mainland-plus-northeast outline for grid masking. A detailed GIS
# boundary can replace this later without changing the grid API.
INDIA_BOUNDARY = (
    (8.0, 77.5),
    (9.0, 76.0),
    (12.0, 74.0),
    (16.0, 72.0),
    (20.0, 68.5),
    (24.0, 68.0),
    (28.0, 70.0),
    (32.0, 74.0),
    (35.0, 74.0),
    (37.0, 76.0),
    (35.0, 79.0),
    (35.0, 82.0),
    (34.0, 84.0),
    (32.0, 87.0),
    (29.0, 88.0),
    (28.0, 92.0),
    (27.0, 97.0),
    (23.0, 97.5),
    (22.0, 94.0),
    (20.0, 92.0),
    (18.0, 89.0),
    (15.0, 86.0),
    (13.0, 82.0),
    (10.0, 80.0),
)


def _axis_values(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= end + step * 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def _point_in_polygon(latitude: float, longitude: float, polygon: Iterable[tuple[float, float]]) -> bool:
    vertices = list(polygon)
    inside = False
    previous_latitude, previous_longitude = vertices[-1]
    for current_latitude, current_longitude in vertices:
        crosses = (current_longitude > longitude) != (previous_longitude > longitude)
        if crosses:
            boundary_latitude = (
                (previous_latitude - current_latitude)
                * (longitude - current_longitude)
                / (previous_longitude - current_longitude)
                + current_latitude
            )
            if latitude < boundary_latitude:
                inside = not inside
        previous_latitude, previous_longitude = current_latitude, current_longitude
    return inside


def is_in_india(latitude: float, longitude: float) -> bool:
    """Return whether a point is inside the compact India mask."""
    return _point_in_polygon(float(latitude), float(longitude), INDIA_BOUNDARY)


def nearest_location(latitude: float, longitude: float) -> str:
    """Return the configured location nearest to a grid cell."""
    return min(
        LOCATION_METADATA,
        key=lambda city: (
            (LOCATION_METADATA[city]["latitude"] - latitude) ** 2
            + (LOCATION_METADATA[city]["longitude"] - longitude) ** 2
        ),
    )


def generate_india_grid(
    latitude_resolution: float = DEFAULT_RESOLUTION,
    longitude_resolution: float | None = None,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
) -> list[dict[str, float | str]]:
    """Generate masked grid cells as latitude/longitude/region records."""
    if latitude_resolution <= 0:
        raise ValueError("latitude_resolution must be positive")
    longitude_step = latitude_resolution if longitude_resolution is None else longitude_resolution
    if longitude_step <= 0:
        raise ValueError("longitude_resolution must be positive")
    latitude_min, latitude_max, longitude_min, longitude_max = bounds
    if latitude_min >= latitude_max or longitude_min >= longitude_max:
        raise ValueError("bounds must be ordered as min, max latitude and longitude")

    cells: list[dict[str, float | str]] = []
    for latitude in _axis_values(latitude_min, latitude_max, latitude_resolution):
        for longitude in _axis_values(longitude_min, longitude_max, longitude_step):
            if not is_in_india(latitude, longitude):
                continue
            city = nearest_location(latitude, longitude)
            cells.append(
                {
                    "lat": latitude,
                    "lon": longitude,
                    "region": str(LOCATION_METADATA[city]["region"]),
                    "nearest_city": city,
                }
            )
    return cells
