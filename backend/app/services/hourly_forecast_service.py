from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.forecast_service import (
    REPO_ROOT,
    WEIGHT_MAP_PATH,
    _api_variable,
    validate_variable,
)
from ml.blending.weight_service import get_forecast_weights
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets, extract_gefs_point
from ml.preprocessing.gfs_extract import extract_gfs_point

SUPPORTED_LEADS = (24, 48, 72)
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"
UNITS = {
    "temperature": "°C",
    "rainfall": "mm",
    "wind_speed": "m/s",
}

GEFS_INTERVAL_HOURS = 3


def validate_hourly_request(variable: str, lead_hours: int) -> None:
    validate_variable(variable)
    if lead_hours not in SUPPORTED_LEADS:
        raise ValueError("lead_hours must be one of 24, 48, or 72")


def _cycle_candidates() -> list[datetime]:
    now = datetime.now(timezone.utc)
    candidates = []
    for cycle in (18, 12, 6, 0):
        cycle_time = now.replace(
            hour=cycle,
            minute=0,
            second=0,
            microsecond=0,
        )
        if cycle_time > now:
            cycle_time -= timedelta(days=1)
        candidates.append(cycle_time)
    return candidates


def _download_gfs_hour(forecast_hour: int):
    last_error: Exception | None = None
    for cycle_time in _cycle_candidates():
        try:
            return download_gfs_subsets(
                date=cycle_time,
                forecast_hour=forecast_hour,
                save_dir=REPO_ROOT / "data" / "raw" / "gfs",
            )
        except Exception as exc:
            last_error = exc
    return None


def _download_gefs_hour(forecast_hour: int):
    last_error: Exception | None = None
    for cycle_time in _cycle_candidates():
        try:
            paths = download_gefs_subsets(
                date=cycle_time,
                forecast_hour=forecast_hour,
                save_dir=REPO_ROOT / "data" / "raw" / "gefs",
            )
            if paths is None:
                continue
            return paths
        except Exception as exc:
            last_error = exc
    return None


def _blend_value(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    gfs_value: float,
    gefs_value: float,
) -> float:
    weights = get_forecast_weights(
        latitude=latitude,
        longitude=longitude,
        variable=variable,
        lead_hours=lead_hours,
        path=WEIGHT_MAP_PATH,
    )
    return float(
        weights["weights"]["gfs"] * gfs_value
        + weights["weights"]["gefs"] * gefs_value
    )


def _interpolate(values: dict[int, float], forecast_hours: range) -> dict[int, float]:
    available_hours = np.asarray(sorted(values), dtype=float)
    available_values = np.asarray(
        [values[int(hour)] for hour in available_hours],
        dtype=float,
    )
    interpolated = np.interp(
        np.asarray(list(forecast_hours), dtype=float),
        available_hours,
        available_values,
    )
    return {
        hour: float(value)
        for hour, value in zip(forecast_hours, interpolated)
    }


def _hourly_accumulations(cumulative: dict[int, float], lead_hours: int) -> dict[int, float]:
    hourly = {0: 0.0}
    previous = 0.0
    for hour in range(1, lead_hours + 1):
        current = cumulative[hour]
        hourly[hour] = max(current - previous, 0.0)
        previous = current
    return hourly


def _interval_accumulations(cumulative: dict[int, float]) -> dict[int, float]:
    hourly = {0: 0.0}
    previous_hour = 0
    previous_value = 0.0
    for hour in sorted(cumulative):
        hourly[hour] = max(cumulative[hour] - previous_value, 0.0)
        previous_hour = hour
        previous_value = cumulative[hour]
    return hourly


def _valid_value(value: object) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)


def _response(
    variable: str,
    lead_hours: int,
    points: list[dict[str, float]],
    source: str,
) -> dict[str, object]:
    return {
        "variable": variable,
        "lead_hours": lead_hours,
        "unit": UNITS[variable],
        "points": points,
        "source": source,
        "sources": {
            "gfs": source in ("wcast_blend", "gfs_fallback"),
            "gefs": source in ("wcast_blend", "gefs_fallback"),
        },
    }


def _collect_gfs_values(
    latitude: float,
    longitude: float,
    forecast_hours: range,
    field: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for forecast_hour in forecast_hours:
        try:
            extracted = extract_gfs_point(
                _download_gfs_hour(forecast_hour),
                latitude,
                longitude,
                forecast_hour,
            )
            value = extracted.get(field)
            if _valid_value(value):
                values[forecast_hour] = float(value)
        except Exception:
            continue
    return values


def _collect_gefs_values(
    latitude: float,
    longitude: float,
    forecast_hours: range,
    field: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for forecast_hour in forecast_hours:
        try:
            extracted = extract_gefs_point(
                _download_gefs_hour(forecast_hour),
                latitude,
                longitude,
                forecast_hour,
            )
            value = extracted.get(field)
            if _valid_value(value):
                values[forecast_hour] = float(value)
        except Exception:
            continue
    return values


def _generate_hourly_rainfall(
    latitude: float,
    longitude: float,
    lead_hours: int,
) -> dict[str, object]:
    gfs_hours = range(1, lead_hours + 1)
    gefs_hours = range(GEFS_INTERVAL_HOURS, lead_hours + 1, GEFS_INTERVAL_HOURS)

    gfs_cumulative = _collect_gfs_values(
        latitude,
        longitude,
        gfs_hours,
        "precipitation_mm",
    )
    gefs_cumulative = _collect_gefs_values(
        latitude,
        longitude,
        gefs_hours,
        "precipitation_mm",
    )

    gfs_available = len(gfs_cumulative) == len(list(gfs_hours))
    gefs_available = len(gefs_cumulative) == len(list(gefs_hours))

    if not gfs_available and not gefs_available:
        raise RuntimeError(
            f"Neither GFS nor GEFS rainfall data is available for F{lead_hours:03d}"
        )

    gfs_rainfall = _hourly_accumulations(gfs_cumulative, lead_hours) if gfs_available else None
    gefs_rainfall = None
    if gefs_available:
        gefs_rainfall = _interpolate(
            _interval_accumulations(gefs_cumulative),
            range(lead_hours + 1),
        )

    if gfs_available and gefs_available:
        source = "wcast_blend"
        points = [
            {
                "hour": hour,
                "value": _blend_value(
                    latitude,
                    longitude,
                    "precipitation",
                    lead_hours,
                    gfs_rainfall[hour],
                    gefs_rainfall[hour],
                ),
            }
            for hour in range(lead_hours + 1)
        ]
    elif gfs_available:
        source = "gfs_fallback"
        points = [
            {"hour": hour, "value": gfs_rainfall[hour]}
            for hour in range(lead_hours + 1)
        ]
    else:
        source = "gefs_fallback"
        points = [
            {"hour": hour, "value": gefs_rainfall[hour]}
            for hour in range(lead_hours + 1)
        ]

    return _response("rainfall", lead_hours, points, source)


def generate_hourly_forecast(
    latitude: float,
    longitude: float,
    lead_hours: int,
    variable: str,
) -> dict[str, object]:
    validate_hourly_request(variable, lead_hours)
    if variable == "rainfall":
        return _generate_hourly_rainfall(latitude, longitude, lead_hours)

    forecast_hours = range(lead_hours + 1)
    gefs_hours = range(0, lead_hours + 1, GEFS_INTERVAL_HOURS)
    internal_variable = _api_variable(variable)
    gfs_values: dict[str, dict[int, float]] = {
        "temperature": {},
        "precipitation": {},
        "wind_speed": {},
    }
    gefs_values: dict[str, dict[int, float]] = {
        "temperature": {},
        "precipitation": {},
        "wind_speed": {},
    }

    field = {
        "temperature": "temperature_C",
        "precipitation": "precipitation_mm",
        "wind_speed": "wind_speed_ms",
    }[internal_variable]
    gfs_values[internal_variable] = _collect_gfs_values(
        latitude,
        longitude,
        forecast_hours,
        field,
    )
    gefs_values[internal_variable] = _collect_gefs_values(
        latitude,
        longitude,
        gefs_hours,
        field,
    )
    gfs_available = len(gfs_values[internal_variable]) == len(list(forecast_hours))
    gefs_available = len(gefs_values[internal_variable]) == len(list(gefs_hours))

    if not gfs_available and not gefs_available:
        raise RuntimeError(
            f"Neither GFS nor GEFS {variable} data is available for F{lead_hours:03d}"
        )

    interpolated_gefs = _interpolate(
        gefs_values[internal_variable],
        forecast_hours,
    ) if gefs_available else None

    if gfs_available and gefs_available:
        source = "wcast_blend"
        points = [
            {
                "hour": forecast_hour,
                "value": _blend_value(
                    latitude,
                    longitude,
                    internal_variable,
                    lead_hours,
                    gfs_values[internal_variable][forecast_hour],
                    interpolated_gefs[forecast_hour],
                ),
            }
            for forecast_hour in forecast_hours
        ]
    elif gfs_available:
        source = "gfs_fallback"
        points = [
            {"hour": hour, "value": gfs_values[internal_variable][hour]}
            for hour in forecast_hours
        ]
    else:
        source = "gefs_fallback"
        points = [
            {"hour": hour, "value": interpolated_gefs[hour]}
            for hour in forecast_hours
        ]

    return _response(variable, lead_hours, points, source)
