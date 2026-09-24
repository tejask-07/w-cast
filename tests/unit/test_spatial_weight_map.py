"""Tests for the India grid and spatial historical skill map."""

import json
from pathlib import Path

import pytest

from ml.spatial.india_grid import generate_india_grid, is_in_india, nearest_location
from ml.spatial.weight_map import build_india_weight_map, summarize_weight_map


SMOKE_DATASET = Path("data/processed/history_2d_20locations_multilead.json")


def _record(city, region, variable, lead, index):
    observation = 10.0 + index
    return {
        "city": city,
        "region": region,
        "variable": variable,
        "lead_hours": lead,
        "observation": observation,
        "gfs": observation + 1.0,
        "gefs": observation + 2.0,
    }


def test_default_grid_generation_and_boundary_filtering():
    grid = generate_india_grid()

    assert grid
    assert len(grid) > 100
    assert all(is_in_india(cell["lat"], cell["lon"]) for cell in grid)
    assert is_in_india(19.076, 72.8777)
    assert not is_in_india(0.0, 80.0)
    assert not is_in_india(40.0, 80.0)


def test_configurable_resolution_reduces_cell_count():
    coarse = generate_india_grid(latitude_resolution=1.0, longitude_resolution=1.0)
    fine = generate_india_grid(latitude_resolution=0.5, longitude_resolution=0.5)

    assert len(fine) > len(coarse)
    assert all(round(cell["lat"] * 2) == cell["lat"] * 2 for cell in fine)


def test_region_assignment_reuses_location_metadata():
    assert nearest_location(19.076, 72.8777) == "Mumbai"
    cell = next(
        cell
        for cell in generate_india_grid(
            latitude_resolution=0.5,
            longitude_resolution=0.5,
            bounds=(19.0, 19.5, 72.5, 73.0),
        )
        if cell["lat"] == 19.0 and cell["lon"] == 72.5
    )
    assert cell["region"] == "west_coast"


def test_smoke_data_falls_back_to_india_and_covers_all_outputs():
    payload = json.loads(SMOKE_DATASET.read_text(encoding="utf-8"))
    weight_map = build_india_weight_map(
        payload["records"],
        latitude_resolution=1.0,
        longitude_resolution=1.0,
        bounds=(18.0, 19.0, 72.0, 74.0),
    )

    assert weight_map
    assert {cell["skill_source"] for cell in weight_map} == {"india"}
    for cell in weight_map:
        assert set(cell) >= {"lat", "lon", "region", "skill_source"}
        for variable in ("temperature", "precipitation", "wind_speed"):
            assert set(cell[variable]) == {"24", "48", "72"}
            for lead in ("24", "48", "72"):
                skill = cell[variable][lead]
                assert skill["gefs_mae"] is None
                assert sum(skill["weights"].values()) == pytest.approx(1.0)


def test_city_skill_assignment_and_weight_normalization():
    records = [
        _record("Mumbai", "west_coast", variable, lead, index)
        for variable in ("temperature", "precipitation", "wind_speed")
        for lead in (24, 48, 72)
        for index in range(3)
    ]
    weight_map = build_india_weight_map(
        records,
        latitude_resolution=1.0,
        longitude_resolution=1.0,
        bounds=(19.0, 19.5, 72.0, 73.0),
    )

    assert weight_map
    assert {cell["skill_source"] for cell in weight_map} == {"city"}
    assert all(
        sum(cell[variable][str(lead)]["weights"].values()) == pytest.approx(1.0)
        for cell in weight_map
        for variable in ("temperature", "precipitation", "wind_speed")
        for lead in (24, 48, 72)
    )


def test_region_skill_assignment_and_summary_counts():
    records = [
        _record("Pune", "west", "temperature", 24, index)
        for index in range(2)
    ] + [_record("Ahmedabad", "west", "temperature", 24, 2)]
    weight_map = build_india_weight_map(
        records,
        latitude_resolution=1.0,
        longitude_resolution=1.0,
        bounds=(18.5, 19.0, 73.5, 74.0),
        variables=("temperature",),
        leads=(24,),
    )

    assert weight_map
    assert {cell["skill_source"] for cell in weight_map} == {"region"}
    summary = summarize_weight_map(weight_map)
    assert summary["region_skill_cells"] == len(weight_map)
    assert summary["city_skill_cells"] == 0
