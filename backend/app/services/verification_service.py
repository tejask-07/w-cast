from pathlib import Path
from typing import Dict
import json

import numpy as np


SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")

VARIABLE_MAP = {
    "temperature": "temperature",
    "rainfall": "precipitation",
    "wind_speed": "wind_speed",
}


def validate_variable(variable: str) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")


def _nearest_city(lat: float, lon: float) -> str:
    cities = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
    }

    return min(
        cities,
        key=lambda city: (
            (cities[city][0] - lat) ** 2
            + (cities[city][1] - lon) ** 2
        ),
    )


def _load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["records"]


def _calculate_metrics(records: list[dict]) -> dict:
    observations = np.asarray(
        [float(record["observation"]) for record in records],
        dtype=float,
    )

    gfs = np.asarray(
        [float(record["gfs"]) for record in records],
        dtype=float,
    )

    gefs = np.asarray(
        [float(record["gefs"]) for record in records],
        dtype=float,
    )

    gfs_error = gfs - observations
    gefs_error = gefs - observations

    gfs_mae = np.mean(np.abs(gfs_error))
    gefs_mae = np.mean(np.abs(gefs_error))

    gfs_rmse = np.sqrt(np.mean(gfs_error ** 2))
    gefs_rmse = np.sqrt(np.mean(gefs_error ** 2))

    gfs_bias = np.mean(gfs_error)
    gefs_bias = np.mean(gefs_error)

    gfs_score = 1.0 / (gfs_mae + 1e-6)
    gefs_score = 1.0 / (gefs_mae + 1e-6)

    total = gfs_score + gefs_score

    gfs_weight = gfs_score / total
    gefs_weight = gefs_score / total

    adaptive_blend = (
        gfs_weight * gfs
        + gefs_weight * gefs
    )

    blend_error = adaptive_blend - observations

    blend_mae = np.mean(np.abs(blend_error))
    blend_rmse = np.sqrt(np.mean(blend_error ** 2))
    blend_bias = np.mean(blend_error)

    return {
        "gfs": {
            "mae": float(gfs_mae),
            "rmse": float(gfs_rmse),
            "bias": float(gfs_bias),
        },
        "gefs": {
            "mae": float(gefs_mae),
            "rmse": float(gefs_rmse),
            "bias": float(gefs_bias),
        },
        "adaptive_blend": {
            "mae": float(blend_mae),
            "rmse": float(blend_rmse),
            "bias": float(blend_bias),
        },
    }


def generate_verification(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
) -> Dict[str, object]:

    validate_variable(variable)

    city = _nearest_city(lat, lon)
    internal_variable = VARIABLE_MAP[variable]

    history_path = Path("data/processed/history_2d.json")

    if not history_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {history_path}"
        )

    records = _load_history(history_path)

    filtered = [
        record
        for record in records
        if record.get("city") == city
        and record.get("variable") == internal_variable
        and record.get("lead_hours") == lead_hours
        and record.get("gfs") is not None
        and record.get("gefs") is not None
        and record.get("observation") is not None
    ]

    if not filtered:
        raise ValueError(
            f"No verification data available for "
            f"{city}, {variable}, {lead_hours}h"
        )

    return {
        "variable": variable,
        "lead_hours": lead_hours,
        "metrics": _calculate_metrics(filtered),
    }