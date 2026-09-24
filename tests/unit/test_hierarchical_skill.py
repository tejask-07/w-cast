"""Tests for hierarchical historical skill aggregation and fallback."""

import json
from pathlib import Path

import pytest

from ml.evaluation.historical_skill import (
    aggregate_hierarchical_skill,
    hierarchical_historical_skill,
    select_hierarchical_skill,
)


SMOKE_DATASET = Path("data/processed/history_2d_20locations_multilead.json")


def _record(city, region, gfs, gefs, observation, variable="temperature", lead=24):
    return {
        "city": city,
        "region": region,
        "variable": variable,
        "lead_hours": lead,
        "gfs": gfs,
        "gefs": gefs,
        "observation": observation,
    }


def _balanced_records(city, region, count, variable="temperature", lead=24):
    return [
        _record(city, region, 10.0 + index, 11.0 + index, 9.0 + index, variable, lead)
        for index in range(count)
    ]


def test_smoke_dataset_metrics_and_missing_gefs_are_model_separated():
    payload = json.loads(SMOKE_DATASET.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload["records"]
        if record["city"] == "Mumbai"
        and record["variable"] == "temperature"
        and record["lead_hours"] == 24
    ]

    aggregates = aggregate_hierarchical_skill(records)
    skill = aggregates["city"]["Mumbai"]["temperature"]["24"]

    assert skill["gfs"]["sample_count"] == 2
    assert skill["gefs"]["sample_count"] == 0
    assert skill["gefs"]["mae"] is None
    assert skill["gfs"]["mae"] == pytest.approx(
        sum(record["gfs_absolute_error"] for record in records) / 2
    )


def test_city_level_selection():
    records = _balanced_records("Mumbai", "west_coast", 3)

    result = hierarchical_historical_skill(records, "Mumbai", "temperature", 24)

    assert result["source"] == "city"
    assert result["region"] == "west_coast"
    assert result["sample_count"] == {"gfs": 3, "gefs": 3}


def test_region_fallback_when_city_is_insufficient():
    records = _balanced_records("Pune", "west", 2)
    records += _balanced_records("Ahmedabad", "west", 1)

    result = hierarchical_historical_skill(records, "Pune", "temperature", 24)

    assert result["source"] == "region"
    assert result["region"] == "west"
    assert result["sample_count"] == {"gfs": 3, "gefs": 3}


def test_india_fallback_when_region_is_insufficient():
    records = _balanced_records("Mumbai", "west_coast", 2)
    records += _balanced_records("Jaipur", "northwest", 2)

    result = hierarchical_historical_skill(records, "Mumbai", "temperature", 24)

    assert result["source"] == "india"
    assert result["sample_count"] == {"gfs": 4, "gefs": 4}


def test_missing_gefs_is_not_zero_error():
    records = _balanced_records("Mumbai", "west_coast", 3)
    for record in records:
        record["gefs"] = None

    result = hierarchical_historical_skill(records, "Mumbai", "temperature", 24)

    assert result["source"] == "india"
    assert result["gefs"]["sample_count"] == 0
    assert result["gefs"]["mae"] is None


def test_insufficient_samples_falls_to_india_with_provenance():
    records = _balanced_records("Mumbai", "west_coast", 2)

    result = hierarchical_historical_skill(records, "Mumbai", "temperature", 24)

    assert result["source"] == "india"
    assert result["sample_count"] == {"gfs": 2, "gefs": 2}
    assert result["city"] == "Mumbai"


def test_metric_calculation_and_india_selection():
    records = [
        _record("Mumbai", "west_coast", 12.0, 10.0, 10.0),
        _record("Jaipur", "northwest", 14.0, 8.0, 10.0),
        _record("Delhi", "north", 10.0, 12.0, 10.0),
    ]

    aggregates = aggregate_hierarchical_skill(records)
    result = select_hierarchical_skill(
        aggregates,
        "Mumbai",
        "temperature",
        24,
        minimum_samples=1,
    )

    assert result["source"] == "city"
    assert result["gfs"] == {
        "mae": 2.0,
        "rmse": 2.0,
        "bias": 2.0,
        "sample_count": 1,
    }
    assert result["gefs"]["mae"] == 0.0


def test_smoke_dataset_uses_india_fallback_for_missing_gefs():
    payload = json.loads(SMOKE_DATASET.read_text(encoding="utf-8"))
    result = hierarchical_historical_skill(
        payload["records"],
        "Mumbai",
        "precipitation",
        72,
    )

    assert result["source"] == "india"
    assert result["sample_count"]["gfs"] == 40
    assert result["sample_count"]["gefs"] == 0
