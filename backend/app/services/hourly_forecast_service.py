from __future__ import annotations

from app.services.forecast_service import (
    REPO_ROOT,
    _api_variable,
    _download_forecast_sources,
    validate_variable,
)
from ml.pipeline import generate_forecast as generate_forecast_ml

SUPPORTED_LEADS = (24, 48, 72)
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"
UNITS = {
    "temperature": "°C",
    "rainfall": "mm",
    "wind_speed": "m/s",
}


def validate_hourly_request(variable: str, lead_hours: int) -> None:
    validate_variable(variable)
    if lead_hours not in SUPPORTED_LEADS:
        raise ValueError("lead_hours must be one of 24, 48, or 72")


def generate_hourly_forecast(
    latitude: float,
    longitude: float,
    lead_hours: int,
    variable: str,
) -> dict[str, object]:
    validate_hourly_request(variable, lead_hours)
    internal_variable = _api_variable(variable)
    points = []

    for forecast_hour in range(lead_hours + 1):
        gfs_paths, gefs_paths = _download_forecast_sources(forecast_hour)
        result = generate_forecast_ml(
            lat=latitude,
            lon=longitude,
            lead_hours=forecast_hour,
            variable=internal_variable,
            gfs_file_path=gfs_paths,
            gefs_file_paths=gefs_paths,
            history_path=HISTORY_PATH,
        )
        points.append({
            "hour": forecast_hour,
            "value": float(result["forecast"][internal_variable]),
        })

    return {
        "variable": variable,
        "lead_hours": lead_hours,
        "unit": UNITS[variable],
        "points": points,
    }
