"""MVP forecast pipeline boundary."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ForecastDataUnavailable(RuntimeError):
    """Raised when required forecast sources or observations are not available."""


@dataclass(frozen=True)
class ForecastRequest:
    latitude: float
    longitude: float
    lead_hours: int
    variable: str


def generate_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
    gfs_file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a real GFS point result when a file is supplied.

    GEFS, observations, precipitation, and blended values remain ``None``
    until real sources are configured. With no GFS path this function raises
    :class:`ForecastDataUnavailable` rather than fabricating a forecast.
    """
    if not variable:
        raise ValueError("variable is required")
    if lead_hours < 0:
        raise ValueError("lead_hours must be non-negative")
    request = ForecastRequest(lat, lon, lead_hours, variable)
    if gfs_file_path is None:
        raise ForecastDataUnavailable(
            "A GFS file is required; GEFS, observations, and blended values "
            "are not available and no weather value was fabricated."
        )

    from ml.preprocessing.gfs_extract import extract_gfs_point

    gfs_point = extract_gfs_point(gfs_file_path, lat, lon, lead_hours)
    if variable not in gfs_point:
        raise ValueError(f"Unsupported extracted GFS variable: {variable}")
    return {
        "request": request,
        "gfs": gfs_point,
        "forecast_value": gfs_point[variable],
        "gefs": None,
        "observations": None,
        "equal_weight_blend": None,
        "static_weight_blend": None,
        "adaptive_blend": None,
        "precipitation_mm": None,
    }