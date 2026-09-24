from __future__ import annotations

from typing import Any

from ml.spatial.lookup import get_spatial_weights


def spatial_blend(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    gfs_value: float,
    gefs_value: float,
) -> dict[str, Any]:

    skill = get_spatial_weights(
        latitude=latitude,
        longitude=longitude,
        variable=variable,
        lead_hours=lead_hours,
    )

    weights = skill["weights"]

    blended = (
        weights["gfs"] * gfs_value
        + weights["gefs"] * gefs_value
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "variable": variable,
        "lead_hours": lead_hours,
        "gfs": gfs_value,
        "gefs": gefs_value,
        "weights": weights,
        "blended": blended,
        "skill_source": skill["skill_source"],
        "gfs_mae": skill["gfs_mae"],
        "gefs_mae": skill["gefs_mae"],
        "gfs_sample_count": skill["gfs_sample_count"],
        "gefs_sample_count": skill["gefs_sample_count"],
    }
