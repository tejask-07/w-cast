from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict

from app.services.gfs_forecast_service import (
    generate_real_gfs_forecast as _generate_real_gfs_forecast,
)

from ml.blending.blender import blend_forecasts
from ml.evaluation.historical_weights import build_historical_weights
from ml.pipeline import generate_forecast as generate_forecast_ml
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets
from ml.regimes.classifier import classify_regime
from ml.regimes.extremes import detect_extremes


SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")
REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = REPO_ROOT / "data" / "processed" / "history_2d.json"
HISTORY_VARIABLES = {
    "temperature": "temperature",
    "rainfall": "precipitation",
    "wind_speed": "wind_speed",
}


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


def _load_historical_records() -> list[dict]:
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)["records"]


def _historical_metadata(
    lat: float,
    lon: float,
    lead_hours: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if lead_hours <= 0:
        raise ValueError("lead_hours must be greater than zero")

    city = resolve_location_name(lat, lon)
    historical_weights = build_historical_weights(HISTORY_PATH)
    weights = historical_weights.get(city, {}).get("temperature", {}).get(
        str(lead_hours), {}
    ).get("weights", {"gfs": 0.5, "gefs": 0.5})

    records = {
        record["variable"]: record
        for record in _load_historical_records()
        if record.get("city") == city
        and record.get("lead_hours") == lead_hours
        and record.get("variable") in HISTORY_VARIABLES.values()
        and record.get("gfs") is not None
        and record.get("gefs") is not None
    }

    missing = set(HISTORY_VARIABLES.values()) - set(records)
    if missing:
        raise ValueError(
            f"No historical forecast data available for {city}, {lead_hours}h"
        )

    values = {
        variable: float(
            blend_forecasts(
                {"gfs": records[internal]["gfs"], "gefs": records[internal]["gefs"]},
                weights,
            )
        )
        for variable, internal in HISTORY_VARIABLES.items()
    }

    return weights, values


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
    history_path = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"

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
        history_path=history_path,
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

    weights, values = _historical_metadata(lat, lon, lead_hours)

    return {
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "weights": {
            "gfs": weights["gfs"],
            "gefs": weights["gefs"],
            "baseline": 0.0,
        },
        "regime": classify_regime(values["rainfall"]),
    }


def generate_extremes(
    lat: float,
    lon: float,
    lead_hours: int,
) -> Dict[str, object]:

    _, values = _historical_metadata(lat, lon, lead_hours)
    extremes = detect_extremes(
        precipitation_mm=values["rainfall"],
        temperature_c=values["temperature"],
        wind_speed_ms=values["wind_speed"],
    )

    return {
        **extremes,
        "risk_level": _risk_level(extremes),
    }