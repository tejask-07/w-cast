from datetime import datetime, timezone

import pytest

from ml.blending.weight_service import get_forecast_weights
from ml.evaluation.historical_skill import (
    aggregate_contextual_hierarchical_skill,
    build_verification_record,
    record_context,
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


def test_get_forecast_weights_respects_context_over_spatial_baseline(monkeypatch, tmp_path):
    weight_map = [{
        "lat": 19.076,
        "lon": 72.878,
        "region": "west_coast",
        "temperature": {
            "24": {
                "skill_source": "global_baseline",
                "weights": {"gfs": 0.7, "gefs": 0.3},
                "gfs_mae": 1.0,
                "gefs_mae": 1.0,
                "gfs_sample_count": 10,
                "gefs_sample_count": 10,
            }
        },
    }]
    path = tmp_path / "weights.json"
    path.write_text(__import__("json").dumps(weight_map), encoding="utf-8")

    monkeypatch.setattr(
        "ml.blending.weight_service._load_historical_context_records",
        lambda: [{
            "city": "Mumbai",
            "region": "west_coast",
            "variable": "temperature",
            "lead_hours": 24,
            "valid_time": "2026-07-01T00:00:00Z",
            "gfs": 31.0,
            "gefs": 24.0,
            "observation": 25.0,
            "forecast_precipitation_mm": 0.5,
        }],
    )

    result = get_forecast_weights(
        latitude=19.076,
        longitude=72.878,
        variable="temperature",
        lead_hours=24,
        season="monsoon",
        regime="DRY",
        path=path,
    )

    assert result["weights"]["gefs"] > result["weights"]["gfs"]
    assert result["season"] == "monsoon"
    assert result["regime"] == "DRY"


def test_get_forecast_weights_loads_dict_history_records_and_respects_regime(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(__import__("json").dumps({
        "records": [
            {
                "city": "Mumbai",
                "region": "west_coast",
                "variable": "precipitation",
                "lead_hours": 24,
                "valid_time": "2026-07-01T00:00:00Z",
                "gfs": 20.0,
                "gefs": 10.0,
                "observation": 10.0,
                "forecast_precipitation_mm": 0.5,
                "season": "monsoon",
                "regime": "DRY",
            },
            {
                "city": "Mumbai",
                "region": "west_coast",
                "variable": "precipitation",
                "lead_hours": 24,
                "valid_time": "2026-07-02T00:00:00Z",
                "gfs": 10.0,
                "gefs": 20.0,
                "observation": 10.0,
                "forecast_precipitation_mm": 80.0,
                "season": "monsoon",
                "regime": "WET",
            },
        ]
    }), encoding="utf-8")

    dry = get_forecast_weights(
        latitude=19.076,
        longitude=72.8777,
        variable="precipitation",
        lead_hours=24,
        season="monsoon",
        regime="DRY",
        historical_path=history_path,
    )
    wet = get_forecast_weights(
        latitude=19.076,
        longitude=72.8777,
        variable="precipitation",
        lead_hours=24,
        season="monsoon",
        regime="WET",
        historical_path=history_path,
    )

    assert dry["skill_source"].endswith("DRY")
    assert wet["skill_source"].endswith("WET")
    assert dry["weights"]["gefs"] > dry["weights"]["gfs"]
    assert wet["weights"]["gfs"] > wet["weights"]["gefs"]


def test_temperature_regime_uses_same_case_forecast_precipitation():
    forecast = {
        "gfs": {
            "temperature_C": 31.0,
            "wind_speed_ms": 8.0,
            "precipitation_mm": 1.5,
        },
        "gefs": {
            "temperature_C": 30.0,
            "wind_speed_ms": 7.5,
            "precipitation_mm": 2.0,
        },
    }
    record = build_verification_record(
        "2026-07-01T06:00:00Z",
        "2026-07-02T06:00:00Z",
        "Mumbai",
        24,
        "temperature",
        forecast,
        {"temperature_C": 29.0, "precipitation_mm": 9.0, "wind_speed_ms": 6.5},
    )
    assert record["season"] == "monsoon"
    assert record["forecast_precipitation_mm"] == 1.5
    assert record["regime"] == "DRY"
    assert record_context(record) == ("monsoon", "DRY")


def test_wind_regime_uses_same_case_forecast_precipitation():
    forecast = {
        "gfs": {
            "temperature_C": 31.0,
            "wind_speed_ms": 8.0,
            "precipitation_mm": 25.0,
        },
        "gefs": {
            "temperature_C": 30.0,
            "wind_speed_ms": 7.5,
            "precipitation_mm": 20.0,
        },
    }
    record = build_verification_record(
        "2026-07-01T06:00:00Z",
        "2026-07-02T06:00:00Z",
        "Mumbai",
        24,
        "wind_speed",
        forecast,
        {"temperature_C": 29.0, "precipitation_mm": 90.0, "wind_speed_ms": 9.0},
    )
    assert record["forecast_precipitation_mm"] == 25.0
    assert record["regime"] == "WET"
    assert record_context(record) == ("monsoon", "WET")


def test_precipitation_regime_uses_same_case_forecast_precipitation_not_observation():
    forecast = {
        "gfs": {"precipitation_mm": 60.0},
        "gefs": {"precipitation_mm": 58.0},
    }
    record = build_verification_record(
        "2026-07-01T06:00:00Z",
        "2026-07-02T06:00:00Z",
        "Mumbai",
        24,
        "precipitation",
        forecast,
        {"precipitation_mm": 120.0},
    )
    assert record["regime"] == "WET"
    assert record_context(record)[1] == "WET"


def test_record_context_prefers_explicit_context_and_falls_back_safely():
    explicit = {
        "valid_time": "2026-07-02T06:00:00Z",
        "season": "monsoon",
        "regime": "DRY",
        "variable": "temperature",
        "forecast_precipitation_mm": 0.0,
    }
    assert record_context(explicit) == ("monsoon", "DRY")

    legacy = {
        "valid_time": "2026-07-02T06:00:00Z",
        "variable": "temperature",
        "gfs": 31.0,
        "forecast_precipitation_mm": None,
    }
    assert record_context(legacy) == ("monsoon", "UNKNOWN")

    same_case = {
        "season": "winter",
        "regime": "NORMAL",
        "forecast_precipitation_mm": 10.0,
        "variable": "wind_speed",
        "valid_time": "2026-01-05T06:00:00Z",
    }
    assert record_context(same_case) == ("winter", "NORMAL")


def test_same_case_precipitation_context_is_deterministic_across_variables():
    forecast = {
        "gfs": {"temperature_C": 32.0, "wind_speed_ms": 9.0, "precipitation_mm": 12.0},
        "gefs": {"temperature_C": 31.0, "wind_speed_ms": 8.5, "precipitation_mm": 11.0},
    }
    temperature_record = build_verification_record(
        "2026-07-01T06:00:00Z",
        "2026-07-02T06:00:00Z",
        "Mumbai",
        24,
        "temperature",
        forecast,
        {"temperature_C": 30.0, "precipitation_mm": 20.0, "wind_speed_ms": 7.0},
    )
    wind_record = build_verification_record(
        "2026-07-01T06:00:00Z",
        "2026-07-02T06:00:00Z",
        "Mumbai",
        24,
        "wind_speed",
        forecast,
        {"temperature_C": 30.0, "precipitation_mm": 20.0, "wind_speed_ms": 7.0},
    )
    assert temperature_record["forecast_precipitation_mm"] == wind_record["forecast_precipitation_mm"] == 12.0
    assert temperature_record["regime"] == wind_record["regime"] == "NORMAL"


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
