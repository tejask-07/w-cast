from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.forecast_service import (
    REPO_ROOT,
    WEIGHT_MAP_PATH,
    _api_variable,
    validate_variable,
)
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets, extract_gefs_point
from ml.preprocessing.gfs_extract import extract_gfs_point
from ml.spatial.lookup import get_spatial_weights

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
                save_dir="data/raw/gfs",
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"No available GFS forecast cycle found for F{forecast_hour:03d}"
    ) from last_error


def _download_gefs_hour(forecast_hour: int):
    last_error: Exception | None = None
    for cycle_time in _cycle_candidates():
        try:
            paths = download_gefs_subsets(
                date=cycle_time,
                forecast_hour=forecast_hour,
                save_dir="data/raw/gefs",
            )
            if paths is None:
                continue
            return paths
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"No available GEFS ensemble-mean forecast cycle found for F{forecast_hour:03d}"
    ) from last_error


def _blend_value(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    gfs_value: float,
    gefs_value: float,
) -> float:
    weights = get_spatial_weights(
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


def _generate_hourly_rainfall(
    latitude: float,
    longitude: float,
    lead_hours: int,
) -> dict[str, object]:
    gfs_cumulative: dict[int, float] = {}
    for forecast_hour in range(lead_hours + 1):
        extracted = extract_gfs_point(
            _download_gfs_hour(forecast_hour),
            latitude,
            longitude,
            forecast_hour,
        )
        if forecast_hour == 0:
            continue
        value = extracted.get("precipitation_mm")
        if value is None:
            raise RuntimeError(
                f"GFS field unavailable for rainfall at F{forecast_hour:03d}"
            )
        gfs_cumulative[forecast_hour] = float(value)

    gfs_rainfall = _hourly_accumulations(gfs_cumulative, lead_hours)
    gefs_cumulative: dict[int, float] = {}
    for forecast_hour in range(0, lead_hours + 1, GEFS_INTERVAL_HOURS):
        extracted = extract_gefs_point(
            _download_gefs_hour(forecast_hour),
            latitude,
            longitude,
            forecast_hour,
        )
        value = extracted.get("precipitation_mm")
        if value is not None and forecast_hour > 0:
            gefs_cumulative[forecast_hour] = float(value)

    if len(gefs_cumulative) == len(range(3, lead_hours + 1, GEFS_INTERVAL_HOURS)):
        gefs_rainfall_at_intervals = _interval_accumulations(gefs_cumulative)
        gefs_rainfall = _interpolate(
            gefs_rainfall_at_intervals,
            range(lead_hours + 1),
        )
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
    else:
        # GEFS APCP is optional in the official ensemble-mean product. Keep
        # rainfall real and explicit by using the GFS accumulation only.
        points = [
            {"hour": hour, "value": gfs_rainfall[hour]}
            for hour in range(lead_hours + 1)
        ]

    return {
        "variable": "rainfall",
        "lead_hours": lead_hours,
        "unit": UNITS["rainfall"],
        "points": points,
    }
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

    for forecast_hour in forecast_hours:
        extracted = extract_gfs_point(
            _download_gfs_hour(forecast_hour),
            latitude,
            longitude,
            forecast_hour,
        )
        field = {
            "temperature": "temperature_C",
            "precipitation": "precipitation_mm",
            "wind_speed": "wind_speed_ms",
        }[internal_variable]
        value = extracted.get(field)
        if value is None:
            raise RuntimeError(
                f"GFS field unavailable for {variable} at F{forecast_hour:03d}"
            )
        gfs_values[internal_variable][forecast_hour] = float(value)

    for forecast_hour in gefs_hours:
        extracted = extract_gefs_point(
            _download_gefs_hour(forecast_hour),
            latitude,
            longitude,
            forecast_hour,
        )
        field = {
            "temperature": "temperature_C",
            "precipitation": "precipitation_mm",
            "wind_speed": "wind_speed_ms",
        }[internal_variable]
        value = extracted.get(field)
        if value is not None:
            gefs_values[internal_variable][forecast_hour] = float(value)

    if not gefs_values[internal_variable]:
        raise RuntimeError(
            f"GEFS field unavailable for {variable} across the requested forecast range"
        )
    if len(gefs_values[internal_variable]) != len(list(gefs_hours)):
        raise RuntimeError(
            f"GEFS field unavailable for {variable} at one or more forecast hours"
        )

    interpolated_gefs = _interpolate(
        gefs_values[internal_variable],
        forecast_hours,
    )
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

    return {
        "variable": variable,
        "lead_hours": lead_hours,
        "unit": UNITS[variable],
        "points": points,
    }
