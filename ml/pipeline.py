
"""Adaptive GFS + GEFS forecast pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.blending.blender import blend_forecasts
from ml.evaluation.historical_weights import build_historical_weights
from ml.regimes.classifier import classify_regime
from ml.regimes.extremes import detect_extremes


class ForecastDataUnavailable(RuntimeError):
    """Raised when required forecast data is unavailable."""


@dataclass(frozen=True)
class ForecastRequest:
    latitude: float
    longitude: float
    lead_hours: int
    variable: str


FIELDS = {
    "temperature": "temperature_C",
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}

CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
}


def _nearest_city(lat, lon):
    return min(
        CITIES,
        key=lambda city: (
            (CITIES[city][0] - lat) ** 2
            + (CITIES[city][1] - lon) ** 2
        ),
    )


def generate_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
    gfs_file_path: str | Path | dict | None = None,
    gefs_file_paths: dict | None = None,
    history_path: str | Path = "data/processed/history_7d_multilead.json",
) -> dict[str, Any]:

    if variable not in FIELDS:
        raise ValueError(f"Unsupported variable: {variable}")

    if lead_hours < 0:
        raise ValueError("lead_hours must be non-negative")

    if gfs_file_path is None or gefs_file_paths is None:
        raise ForecastDataUnavailable(
            "Both GFS and GEFS forecast files are required."
        )

    from ml.preprocessing.gfs_extract import extract_gfs_point
    from ml.preprocessing.gefs import extract_gefs_point

    request = ForecastRequest(lat, lon, lead_hours, variable)

    gfs = extract_gfs_point(
        gfs_file_path, lat, lon, lead_hours
    )

    gefs = extract_gefs_point(
        gefs_file_paths, lat, lon, lead_hours
    )

    city = _nearest_city(lat, lon)
    history = build_historical_weights(history_path)

    blended = {}
    all_weights = {}
    historical_skill = {}

    for name, field in FIELDS.items():
        gfs_value = gfs.get(field)
        gefs_value = gefs.get(field)

        if gfs_value is None or gefs_value is None:
            raise ForecastDataUnavailable(
                f"Missing {name} from GFS or GEFS."
            )

        skill = (
            history.get(city, {})
            .get(name, {})
            .get(str(lead_hours))
        )

        if skill is None:
            weights = {"gfs": 0.5, "gefs": 0.5}
            historical_skill[name] = None
        else:
            weights = skill["weights"]
            historical_skill[name] = {
                "gfs_mae": skill["gfs_mae"],
                "gefs_mae": skill["gefs_mae"],
                "gfs_sample_count": skill["gfs_sample_count"],
                "gefs_sample_count": skill["gefs_sample_count"],
                "source": skill["source"],
            }

        blended[name] = float(
            blend_forecasts(
                {"gfs": gfs_value, "gefs": gefs_value},
                weights,
            )
        )

        all_weights[name] = weights

    regime = classify_regime(blended["precipitation"])

    extremes = detect_extremes(
        precipitation_mm=blended["precipitation"],
        temperature_c=blended["temperature"],
        wind_speed_ms=blended["wind_speed"],
    )

    return {
        "request": request,
        "city": city,
        "gfs": gfs,
        "gefs": gefs,
        "forecast_value": blended[variable],
        "adaptive_blend": blended[variable],
        "weights": all_weights[variable],
        "forecast": blended,
        "all_weights": all_weights,
        "regime": regime,
        "extremes": extremes,
        "historical_skill": historical_skill,
        "precipitation_mm": blended["precipitation"],
    }
