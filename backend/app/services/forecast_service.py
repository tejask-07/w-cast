from __future__ import annotations

from typing import Dict

from app.services.gfs_forecast_service import (
    generate_real_gfs_forecast as _generate_real_gfs_forecast,
    select_gfs_cycle,
)

from ml.pipeline import generate_forecast as generate_forecast_ml
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets


SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")


def resolve_location_name(lat: float, lon: float) -> str:
    location_map = {
        (19.07, 72.87): "Mumbai",
        (28.61, 77.21): "Delhi",
        (22.57, 88.36): "Kolkata",
        (13.08, 80.27): "Chennai",
    }

    return location_map.get(
        (round(lat, 2), round(lon, 2)),
        "Selected Location",
    )


def validate_variable(variable: str) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")


def generate_real_gfs_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
) -> Dict[str, float]:
    return _generate_real_gfs_forecast(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable=variable,
    )


def _download_forecast_sources(lead_hours: int):
    cycle_time = select_gfs_cycle()

    try:
        gfs_paths = download_gfs_subsets(
            date=cycle_time,
            forecast_hour=lead_hours,
            save_dir="data/raw/gfs",
        )

        gefs_paths = download_gefs_subsets(
            date=cycle_time,
            forecast_hour=lead_hours,
            save_dir="data/raw/gefs",
        )
    except Exception as exc:
        raise RuntimeError(
            "GFS or GEFS data unavailable from NOAA NOMADS"
        ) from exc

    return gfs_paths, gefs_paths


def _api_variable(variable: str) -> str:
    if variable == "rainfall":
        return "precipitation"

    return variable


def _risk_level(extremes: dict[str, bool]) -> str:
    count = sum(extremes.values())

    if count >= 2:
        return "high"

    if count == 1:
        return "moderate"

    return "low"


def generate_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
) -> Dict[str, object]:

    validate_variable(variable)

    if lead_hours <= 0:
        raise ValueError("lead_hours must be greater than zero")

    location_name = resolve_location_name(lat, lon)

    gfs_paths, gefs_paths = _download_forecast_sources(lead_hours)

    result = generate_forecast_ml(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable=_api_variable(variable),
        gfs_file_path=gfs_paths,
        gefs_file_paths=gefs_paths,
    )

    weights = result["all_weights"]
    selected_weights = weights[_api_variable(variable)]

    return {
        "location": {
            "name": location_name,
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "forecast": {
            "temperature": result["forecast"]["temperature"],
            "rainfall": result["forecast"]["precipitation"],
            "wind_speed": result["forecast"]["wind_speed"],
        },
        "weights": {
            "gfs": selected_weights["gfs"],
            "gefs": selected_weights["gefs"],
            "baseline": 0.0,
        },
        "regime": result["regime"],
        "extremes": {
            **result["extremes"],
            "risk_level": _risk_level(result["extremes"]),
        },
    }


def generate_weights(
    lat: float,
    lon: float,
    lead_hours: int,
) -> Dict[str, object]:

    result = generate_forecast(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable="temperature",
    )

    return {
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "weights": result["weights"],
        "regime": result["regime"],
    }


def generate_extremes(
    lat: float,
    lon: float,
    lead_hours: int,
) -> Dict[str, object]:

    result = generate_forecast(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable="temperature",
    )

    return result["extremes"]