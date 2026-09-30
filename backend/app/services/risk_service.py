from __future__ import annotations

import numpy as np

from app.services.forecast_service import _download_forecast_sources
from app.services.spatial_forecast_service import (
    _crop_region,
    _interpolate_to_grid,
    _read_model_field,
    _validate_bounds,
    _weights,
)
from ml.blending.blender import blend_forecasts

RISK_GRID_CELLS_PER_AXIS = 12


def calculate_weather_risk(
    temperature: np.ndarray,
    rainfall: np.ndarray,
    wind_speed: np.ndarray,
    temperature_disagreement: np.ndarray,
    rainfall_disagreement: np.ndarray,
    wind_disagreement: np.ndarray,
    lead_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unvalidated 0-100 weather-risk indicator and model confidence."""
    rainfall_threshold = 50.0 * lead_hours / 24
    rainfall_risk = np.clip(rainfall / rainfall_threshold, 0, 1)
    wind_risk = np.clip(wind_speed / 25.0, 0, 1)
    heat_risk = np.clip((temperature - 30.0) / 15.0, 0, 1)
    cold_risk = np.clip((10.0 - temperature) / 15.0, 0, 1)
    temperature_risk = np.maximum(heat_risk, cold_risk)

    uncertainty = np.mean(
        [
            np.clip(temperature_disagreement / 15.0, 0, 1),
            np.clip(rainfall_disagreement / rainfall_threshold, 0, 1),
            np.clip(wind_disagreement / 25.0, 0, 1),
        ],
        axis=0,
    )
    confidence = np.clip(100.0 * (1.0 - uncertainty), 0, 100)
    risk_score = np.clip(
        100.0
        * (
            0.35 * rainfall_risk
            + 0.25 * wind_risk
            + 0.20 * temperature_risk
            + 0.20 * uncertainty
        ),
        0,
        100,
    )
    return risk_score, confidence


def _target_axis(start: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(start, stop, RISK_GRID_CELLS_PER_AXIS + 1)
    return edges, (edges[:-1] + edges[1:]) / 2


def _sample_model_field(
    paths,
    variable: str,
    latitude: float,
    longitude: float,
    bounds: tuple[float, float, float, float],
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
) -> np.ndarray:
    source_latitudes, source_longitudes, values = _read_model_field(paths, variable)
    source_latitudes, source_longitudes, values = _crop_region(
        source_latitudes,
        source_longitudes,
        values,
        latitude,
        longitude,
        bounds,
    )
    return _interpolate_to_grid(
        source_latitudes,
        source_longitudes,
        values,
        grid_latitudes,
        grid_longitudes,
    )


def generate_spatial_risk_grid(
    latitude: float,
    longitude: float,
    lead_hours: int,
    south: float | None = None,
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
) -> dict[str, object]:
    """Build AOI cells from existing blended forecast fields and model spread."""
    bounds = _validate_bounds(south, north, west, east)
    if bounds is None:
        bounds = (
            max(-90.0, latitude - 5.0),
            min(90.0, latitude + 5.0),
            max(-180.0, longitude - 5.0),
            min(180.0, longitude + 5.0),
        )
    if lead_hours not in (24, 48, 72):
        raise ValueError("lead_hours must be one of 24, 48, or 72")

    south, north, west, east = bounds
    latitude_edges, grid_latitudes = _target_axis(south, north)
    longitude_edges, grid_longitudes = _target_axis(west, east)
    gfs_paths, gefs_paths = _download_forecast_sources(lead_hours)
    blended: dict[str, np.ndarray] = {}
    disagreements: dict[str, np.ndarray] = {}

    for variable in ("temperature", "rainfall", "wind_speed"):
        gfs = _sample_model_field(
            gfs_paths,
            variable,
            latitude,
            longitude,
            bounds,
            grid_latitudes,
            grid_longitudes,
        )
        gefs = _sample_model_field(
            gefs_paths,
            variable,
            latitude,
            longitude,
            bounds,
            grid_latitudes,
            grid_longitudes,
        )
        blended[variable] = blend_forecasts(
            {"gfs": gfs, "gefs": gefs},
            _weights(latitude, longitude, variable, lead_hours),
        )
        disagreements[variable] = np.abs(gfs - gefs)

    risk_scores, confidence = calculate_weather_risk(
        blended["temperature"],
        blended["rainfall"],
        blended["wind_speed"],
        disagreements["temperature"],
        disagreements["rainfall"],
        disagreements["wind_speed"],
        lead_hours,
    )
    cells = []
    for row in range(len(grid_latitudes)):
        for column in range(len(grid_longitudes)):
            cells.append({
                "south": float(latitude_edges[row]),
                "north": float(latitude_edges[row + 1]),
                "west": float(longitude_edges[column]),
                "east": float(longitude_edges[column + 1]),
                "temperature": float(blended["temperature"][row, column]),
                "rainfall": float(blended["rainfall"][row, column]),
                "wind_speed": float(blended["wind_speed"][row, column]),
                "risk_score": float(risk_scores[row, column]),
                "confidence": float(confidence[row, column]),
            })

    return {
        "lead_hours": lead_hours,
        "bounds": [[south, west], [north, east]],
        "cells": cells,
        "indicator": "W-CAST weather risk indicator; not a validated hazard probability.",
    }