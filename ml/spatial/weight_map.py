"""Assign hierarchical historical skill and adaptive weights to India cells."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.historical_skill import (
    MIN_LOCAL_SAMPLES,
    aggregate_hierarchical_skill,
    select_hierarchical_skill,
)
from ml.spatial.india_grid import (
    DEFAULT_BOUNDS,
    DEFAULT_RESOLUTION,
    generate_india_grid,
)

DEFAULT_LEADS = (24, 48, 72)
DEFAULT_VARIABLES = ("temperature", "precipitation", "wind_speed")


def _deepest_source(sources: Iterable[str]) -> str:
    ordered = {"city": 0, "region": 1, "india": 2}
    return max(sources, key=lambda source: ordered[source], default="india")


def build_india_weight_map(
    records: list[Mapping[str, Any]],
    latitude_resolution: float = DEFAULT_RESOLUTION,
    longitude_resolution: float | None = None,
    leads: Iterable[int] = DEFAULT_LEADS,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    minimum_samples: int = MIN_LOCAL_SAMPLES,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Build cell weights from existing historical skill records only."""
    lead_list = tuple(leads)
    variable_list = tuple(variables)
    if not lead_list:
        raise ValueError("At least one lead is required")
    if not variable_list:
        raise ValueError("At least one variable is required")
    if any(lead < 0 for lead in lead_list):
        raise ValueError("leads must be non-negative")

    aggregates = aggregate_hierarchical_skill(records)
    grid = generate_india_grid(
        latitude_resolution=latitude_resolution,
        longitude_resolution=longitude_resolution,
        bounds=DEFAULT_BOUNDS if bounds is None else bounds,
    )
    result: list[dict[str, Any]] = []

    for cell in grid:
        cell_skills: dict[str, dict[str, Any]] = {}
        sources: list[str] = []
        for variable in variable_list:
            cell_skills[variable] = {}
            for lead in lead_list:
                skill = select_hierarchical_skill(
                    aggregates,
                    str(cell["nearest_city"]),
                    variable,
                    lead,
                    minimum_samples=minimum_samples,
                )
                sources.append(skill["source"])
                cell_skills[variable][str(lead)] = {
                    "gfs_mae": skill["gfs"]["mae"],
                    "gefs_mae": skill["gefs"]["mae"],
                    "gfs_rmse": skill["gfs"]["rmse"],
                    "gefs_rmse": skill["gefs"]["rmse"],
                    "gfs_bias": skill["gfs"]["bias"],
                    "gefs_bias": skill["gefs"]["bias"],
                    "gfs_sample_count": skill["gfs"]["sample_count"],
                    "gefs_sample_count": skill["gefs"]["sample_count"],
                    "weights": adaptive_weights(
                        {
                            "gfs": skill["gfs"]["mae"],
                            "gefs": skill["gefs"]["mae"],
                        }
                    ),
                    "skill_source": skill["source"],
                }
        source = _deepest_source(sources)
        result.append(
            {
                "lat": cell["lat"],
                "lon": cell["lon"],
                "region": cell["region"],
                "skill_source": source,
                **cell_skills,
            }
        )

    return result


def summarize_weight_map(weight_map: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize cell count and hierarchy provenance."""
    counts = Counter(cell["skill_source"] for cell in weight_map)
    return {
        "grid_cells": len(weight_map),
        "city_skill_cells": counts["city"],
        "region_skill_cells": counts["region"],
        "india_skill_cells": counts["india"],
    }
