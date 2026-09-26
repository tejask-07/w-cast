from datetime import datetime, timezone

import pytest

from ml.evaluation.historical_skill import (
    aggregate_contextual_hierarchical_skill,
    select_contextual_hierarchical_skill,
)
from ml.features.temporal_features import get_season
from ml.spatial.location import resolve_location
from app.services import hourly_forecast_service


@pytest.mark.parametrize(
    ("month", "season"),
    [(1, "winter"), (2, "winter"), (3, "pre_monsoon"), (4, "pre_monsoon"),
     (5, "pre_monsoon"), (6, "monsoon"), (7, "monsoon"), (8, "monsoon"),
     (9, "monsoon"), (10, "post_monsoon"), (11, "post_monsoon"), (12, "winter")],
)
def test_all_months_have_india_season(month, season):
    assert get_season(datetime(2026, month, 1)) == season


def test_configured_location_resolution_uses_all_locations():
    assert resolve_location(13.0827, 80.2707)["city"] == "Chennai"
    assert resolve_location(34.0837, 74.7973)["city"] == "Srinagar"
    assert resolve_location(9.9252, 78.1198)["city"] == "Kochi"


def test_location_resolution_rejects_outside_india():
    with pytest.raises(ValueError, match="outside"):
        resolve_location(0.0, 0.0)


def test_contextual_skill_uses_forecast_context_and_fallback():
    records = []
    for index in range(3):
        records.append({
            "city": "Mumbai",
            "region": "west_coast",
            "variable": "temperature",
            "lead_hours": 24,
            "valid_time": datetime(2026, 7, 1 + index, tzinfo=timezone.utc).isoformat(),
            "gfs": 30.0,
            "gefs": 28.0,
            "observation": 29.0,
            "forecast_precipitation_mm": 1.0,
        })
    aggregates = aggregate_contextual_hierarchical_skill(records)
    selected = select_contextual_hierarchical_skill(
        aggregates, "Mumbai", "temperature", 24, "monsoon", "DRY"
    )
    assert selected["source"] == "city_monsoon_DRY"
    assert selected["gfs"]["sample_count"] == 3
    assert selected["gefs"]["sample_count"] == 3


def test_hourly_falls_back_to_gfs_only(monkeypatch):
    monkeypatch.setattr(hourly_forecast_service, "_download_gfs_hour", lambda _: {})
    monkeypatch.setattr(hourly_forecast_service, "_download_gefs_hour", lambda _: None)
    monkeypatch.setattr(
        hourly_forecast_service,
        "extract_gfs_point",
        lambda *_args: {"temperature_C": 20.0},
    )
    result = hourly_forecast_service.generate_hourly_forecast(19.076, 72.878, 24, "temperature")
    assert result["source"] == "gfs_fallback"
    assert result["sources"] == {"gfs": True, "gefs": False}
    assert len(result["points"]) == 25


def test_hourly_falls_back_to_gefs_only(monkeypatch):
    monkeypatch.setattr(hourly_forecast_service, "_download_gfs_hour", lambda _: None)
    monkeypatch.setattr(hourly_forecast_service, "_download_gefs_hour", lambda _: {})
    monkeypatch.setattr(
        hourly_forecast_service,
        "extract_gefs_point",
        lambda *_args: {"temperature_C": 22.0},
    )
    result = hourly_forecast_service.generate_hourly_forecast(19.076, 72.878, 24, "temperature")
    assert result["source"] == "gefs_fallback"
    assert result["sources"] == {"gfs": False, "gefs": True}
    assert len(result["points"]) == 25


def test_hourly_raises_when_both_models_are_missing(monkeypatch):
    monkeypatch.setattr(hourly_forecast_service, "_download_gfs_hour", lambda _: None)
    monkeypatch.setattr(hourly_forecast_service, "_download_gefs_hour", lambda _: None)
    with pytest.raises(RuntimeError, match="Neither GFS nor GEFS"):
        hourly_forecast_service.generate_hourly_forecast(19.076, 72.878, 24, "temperature")
