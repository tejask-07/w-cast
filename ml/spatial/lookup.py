from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.spatial.india_grid import is_in_india


DEFAULT_WEIGHT_MAP = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "india_weight_map_7d.json"
)


def load_weight_map(
    path: str | Path = DEFAULT_WEIGHT_MAP,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Weight map not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Weight map must contain a list")

    return data


def find_nearest_cell(
    weight_map: list[dict[str, Any]],
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    if not weight_map:
        raise ValueError("Weight map is empty")

    return min(
        weight_map,
        key=lambda cell: (
            (float(cell["lat"]) - latitude) ** 2
            + (float(cell["lon"]) - longitude) ** 2
        ),
    )


def get_spatial_weights(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    path: str | Path = DEFAULT_WEIGHT_MAP,
) -> dict[str, Any]:

    if not is_in_india(latitude, longitude):
        return {
            "latitude": latitude,
            "longitude": longitude,
            "region": "global",
            "skill_source": "global_baseline",
            "weights": {"gfs": 0.5, "gefs": 0.5},
            "gfs_mae": None,
            "gefs_mae": None,
            "gfs_sample_count": 0,
            "gefs_sample_count": 0,
        }

    weight_map = load_weight_map(path)

    cell = find_nearest_cell(
        weight_map,
        latitude,
        longitude,
    )

    variable_data = cell.get(variable)

    if variable_data is None:
        raise ValueError(
            f"Variable not found: {variable}"
        )

    lead_data = variable_data.get(str(lead_hours))

    if lead_data is None:
        raise ValueError(
            f"Lead not found: {lead_hours}"
        )

    return {
        "latitude": cell["lat"],
        "longitude": cell["lon"],
        "region": cell["region"],
        "skill_source": lead_data["skill_source"],
        "weights": lead_data["weights"],
        "gfs_mae": lead_data["gfs_mae"],
        "gefs_mae": lead_data["gefs_mae"],
        "gfs_sample_count": lead_data["gfs_sample_count"],
        "gefs_sample_count": lead_data["gefs_sample_count"],
    }
