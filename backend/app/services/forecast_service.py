from __future__ import annotations

from typing import Dict, Literal

from app.services.blending_service import get_model_weights
from app.services.regime_service import classify_regime

SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")


def resolve_location_name(lat: float, lon: float) -> str:
    location_map = {
        (19.07, 72.87): "Mumbai",
        (28.61, 77.21): "Delhi",
        (22.57, 88.36): "Kolkata",
        (13.08, 80.27): "Chennai",
    }
    return location_map.get((round(lat, 2), round(lon, 2)), "Selected Location")


def validate_variable(variable: str) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")


def generate_forecast(lat: float, lon: float, lead_hours: int, variable: str) -> Dict[str, object]:
    validate_variable(variable)
    location_name = resolve_location_name(lat, lon)
    weights = get_model_weights(lat, lon, lead_hours)
    regime = classify_regime(lat, lon, lead_hours)
    extremes = generate_extremes(lat, lon, lead_hours)

    forecast = {
        "temperature": 29.2,
        "rainfall": 17.5,
        "wind_speed": 18.7,
    }

    if variable == "temperature":
        forecast["temperature"] = round(max(0.0, 29.2 + (lat / 30.0)), 1)
    elif variable == "rainfall":
        forecast["rainfall"] = round(max(0.0, 17.5 + (lead_hours / 10.0)), 1)
    elif variable == "wind_speed":
        forecast["wind_speed"] = round(max(0.0, 18.7 + (lon / 100.0)), 1)

    return {
        "location": {
            "name": location_name,
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "forecast": forecast,
        "weights": weights,
        "regime": regime,
        "extremes": extremes,
    }


def generate_weights(lat: float, lon: float, lead_hours: int) -> Dict[str, object]:
    return {
        "location": {"lat": lat, "lon": lon},
        "lead_hours": lead_hours,
        "weights": get_model_weights(lat, lon, lead_hours),
        "regime": classify_regime(lat, lon, lead_hours),
    }


def generate_extremes(lat: float, lon: float, lead_hours: int) -> Dict[str, object]:
    _ = (lat, lon, lead_hours)
    return {
        "heavy_rain": True,
        "heat_wave": False,
        "high_wind": False,
        "risk_level": "high",
    }
