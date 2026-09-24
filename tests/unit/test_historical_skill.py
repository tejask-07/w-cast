"""Offline tests for historical verification and skill summaries."""

from datetime import datetime, timezone

import pytest

from ml.evaluation.historical_skill import (
    build_verification_record,
    calculate_contextual_skill,
    summarize_historical_skill,
)


def make_record(city="Mumbai", lead_hours=24, variable="temperature", gfs=30.0, gefs=28.0, observation=29.0, precipitation=4.0):
    init = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if variable not in {"temperature", "precipitation", "wind_speed"}:
        return build_verification_record(init, init, city, lead_hours, variable, {}, {})
    field = {"temperature": "temperature_C", "precipitation": "precipitation_mm", "wind_speed": "wind_speed_ms"}[variable]
    forecast_values = {"temperature_C": 30.0, "precipitation_mm": 5.0, "wind_speed_ms": 4.0}
    forecast_values[field] = gfs
    gefs_values = dict(forecast_values)
    gefs_values[field] = gefs
    observation_values = {"temperature_C": 29.0, "precipitation_mm": precipitation, "wind_speed_ms": 3.0}
    observation_values[field] = observation
    return build_verification_record(init, init + __import__("datetime").timedelta(hours=lead_hours), city, lead_hours, variable, {"gfs": forecast_values, "gefs": gefs_values}, observation_values)


def test_correct_absolute_errors():
    record = make_record(gfs=30.0, gefs=27.0, observation=29.0)
    assert record["gfs_error"] == 1.0
    assert record["gefs_error"] == -2.0
    assert record["gfs_absolute_error"] == 1.0
    assert record["gefs_absolute_error"] == 2.0


def test_summary_metrics():
    summary = summarize_historical_skill([make_record(gfs=30.0, gefs=27.0, observation=29.0)])
    gfs = summary[summary.model == "gfs"].iloc[0]
    assert gfs.sample_count == 1
    assert gfs.mae == pytest.approx(1.0)
    assert gfs.rmse == pytest.approx(1.0)
    assert gfs.bias == pytest.approx(1.0)


def test_missing_gfs_and_gefs():
    record = make_record()
    record["gfs"] = None
    record["gefs"] = None
    record["gfs_error"] = record["gfs_absolute_error"] = None
    record["gefs_error"] = record["gefs_absolute_error"] = None
    summary = summarize_historical_skill([record])
    assert summary.empty


@pytest.mark.parametrize("kwargs", [{"city": "Atlantis"}, {"variable": "pressure"}, {"lead_hours": -1}])
def test_invalid_record_inputs(kwargs):
    with pytest.raises(ValueError):
        make_record(**kwargs)


def test_valid_time_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        build_verification_record(datetime(2024, 1, 1), datetime(2024, 1, 3), "Mumbai", 24, "temperature", {"gfs": 1}, {"temperature_C": 1})


def test_contextual_grouping_by_city_lead_and_regime():
    records = [make_record(city="Mumbai", lead_hours=24), make_record(city="Delhi", lead_hours=48)]
    records[0]["precipitation_mm"] = 1.0
    records[1]["precipitation_mm"] = 30.0
    summary = calculate_contextual_skill(records)
    assert set(summary.city) == {"Mumbai", "Delhi"}
    assert set(summary.lead_hours) == {24, 48}
    assert set(summary.regime) == {"DRY", "WET"}


def test_no_fabricated_values():
    record = make_record(gfs=None, gefs=28.0)
    assert record["gfs"] is None
    assert record["gfs_absolute_error"] is None
    assert record["gefs"] == 28.0