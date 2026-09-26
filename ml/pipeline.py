"""Adaptive GFS + GEFS spatial forecast pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.regimes.classifier import classify_regime
from ml.regimes.extremes import detect_extremes
from ml.spatial.location import resolve_location
from ml.spatial.spatial_blend import spatial_blend


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

def _nearest_city(lat: float, lon: float) -> str:
    return resolve_location(lat, lon)["city"]


def generate_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
    gfs_file_path: str | Path | dict | None = None,
    gefs_file_paths: dict | None = None,
    history_path: str | Path = "data/processed/india_weight_map_audited_gefs_2026-09-20_to_2026-09-24.json",
) -> dict[str, Any]:

    if variable not in FIELDS:
        raise ValueError(f"Unsupported variable: {variable}")

    if lead_hours not in (24, 48, 72):
        raise ValueError("lead_hours must be one of: 24, 48, 72")

    if gfs_file_path is None or gefs_file_paths is None:
        raise ForecastDataUnavailable(
            "Both GFS and GEFS forecast files are required."
        )

    from ml.preprocessing.gfs_extract import extract_gfs_point
    from ml.preprocessing.gefs import extract_gefs_point

    request = ForecastRequest(lat, lon, lead_hours, variable)

    gfs = extract_gfs_point(
        gfs_file_path,
        lat,
        lon,
        lead_hours,
    )

    gefs = extract_gefs_point(
        gefs_file_paths,
        lat,
        lon,
        lead_hours,
    )

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

        result = spatial_blend(
            latitude=lat,
            longitude=lon,
            variable=name,
            lead_hours=lead_hours,
            gfs_value=float(gfs_value),
            gefs_value=float(gefs_value),
        )

        blended[name] = result["blended"]
        all_weights[name] = result["weights"]

        historical_skill[name] = {
            "gfs_mae": result["gfs_mae"],
            "gefs_mae": result["gefs_mae"],
            "gfs_sample_count": result["gfs_sample_count"],
            "gefs_sample_count": result["gefs_sample_count"],
            "source": result["skill_source"],
        }

    regime = classify_regime(blended["precipitation"])

    extremes = detect_extremes(
        precipitation_mm=blended["precipitation"],
        temperature_c=blended["temperature"],
        wind_speed_ms=blended["wind_speed"],
    )

    city = _nearest_city(lat, lon)

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
