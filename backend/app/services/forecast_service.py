from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from app.services.gfs_forecast_service import (
    generate_real_gfs_forecast as _generate_real_gfs_forecast,
)

from ml.pipeline import generate_forecast as generate_forecast_ml
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets


SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")


def resolve_location_name(lat: float, lon: float) -> str:
    locations = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
    }

    return min(
        locations,
        key=lambda name: (
            (locations[name][0] - lat) ** 2
            + (locations[name][1] - lon) ** 2
        ),
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
    now = datetime.now(timezone.utc)

    cycles = [18, 12, 6, 0]

    for cycle in cycles:
        cycle_time = now.replace(
            hour=cycle,
            minute=0,
            second=0,
            microsecond=0,
        )

        if cycle_time > now:
            cycle_time = cycle_time.replace(
                day=cycle_time.day - 1
            )

        print(
            f"Trying forecast cycle: "
            f"{cycle_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )

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

            print(
                f"Using forecast cycle: "
                f"{cycle_time.strftime('%Y-%m-%d %H:%M UTC')}"
            )

            return gfs_paths, gefs_paths

        except Exception as exc:
            print(
                f"Cycle unavailable: "
                f"{cycle_time.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({exc})"
            )
            continue

    raise RuntimeError(
        "No available GFS and GEFS forecast cycle found."
    )


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

    gfs_paths, gefs_paths = _download_forecast_sources(
        lead_hours
    )

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