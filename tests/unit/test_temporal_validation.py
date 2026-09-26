from datetime import datetime, timedelta, timezone
import json

import pytest

from ml.evaluation.temporal_validation import evaluate_expanding_window
from ml.preprocessing.align import LOCATION_METADATA

START = datetime(2026, 9, 1, 6, tzinfo=timezone.utc)


def _record(day_offset, city="Mumbai", variable="temperature", lead=24, gfs=10.0, gefs=12.0, obs=9.0):
    location = LOCATION_METADATA[city]
    cycle = START + timedelta(days=day_offset)
    valid = cycle + timedelta(hours=lead)
    return {
        "location_name": city,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "region": location["region"],
        "forecast_date": cycle.date().isoformat(),
        "forecast_cycle": cycle.isoformat().replace("+00:00", "Z"),
        "valid_time": valid.isoformat().replace("+00:00", "Z"),
        "variable": variable,
        "lead_hours": lead,
        "gfs_value": gfs,
        "gefs_value": gefs,
        "observation_value": obs,
        "gfs_source": "fixture GFS",
        "gefs_source": "fixture operational GEFS",
        "gfs_resolution_degrees": 0.25,
        "gefs_resolution_degrees": 0.25,
    }


def test_expanding_window_skips_initial_dates_and_uses_only_prior_dates():
    records = [_record(day, gfs=10 + day, gefs=12 + day, obs=9 + day) for day in range(6)]
    result = evaluate_expanding_window(records, minimum_training_dates=3, existing_map_path=None)

    assert [item["test_date"] for item in result["report"]["skipped_dates"]] == [
        "2026-09-01", "2026-09-02", "2026-09-03",
    ]
    assert len(result["records"]) == 3
    for case in result["records"]:
        assert case["training_date_end"] < case["test_date"]
        assert case["training_date_count"] == 3 + (
            datetime.fromisoformat(case["test_date"]).day - 4
        )
        assert case["gfs_weight"] + case["gefs_weight"] == pytest.approx(1.0)


def test_test_date_values_do_not_leak_into_its_weights():
    records = [_record(day, gfs=10 + day, gefs=14 + day, obs=8 + day) for day in range(5)]
    baseline = evaluate_expanding_window(records, minimum_training_dates=3, existing_map_path=None)
    changed = [dict(record) for record in records]
    for key in ("gfs_value", "gefs_value", "observation_value"):
        changed[3][key] += 10_000
    altered = evaluate_expanding_window(changed, minimum_training_dates=3, existing_map_path=None)

    baseline_case = next(case for case in baseline["records"] if case["test_date"] == "2026-09-04")
    altered_case = next(case for case in altered["records"] if case["test_date"] == "2026-09-04")
    assert baseline_case["gfs_weight"] == altered_case["gfs_weight"]
    assert baseline_case["gefs_weight"] == altered_case["gefs_weight"]
    assert baseline_case["training_date_end"] == "2026-09-03"


def test_expanding_training_window_grows_on_later_test_dates():
    records = [_record(day, gfs=10 + day, gefs=14, obs=8) for day in range(6)]
    result = evaluate_expanding_window(records, minimum_training_dates=3, existing_map_path=None)
    by_date = {case["test_date"]: case for case in result["records"]}
    assert by_date["2026-09-04"]["training_date_start"] == "2026-09-01"
    assert by_date["2026-09-04"]["training_date_end"] == "2026-09-03"
    assert by_date["2026-09-05"]["training_date_end"] == "2026-09-04"
    assert by_date["2026-09-05"]["training_date_count"] == 4


def test_hierarchy_uses_region_then_india_fallback_without_lowering_threshold():
    region_records = [
        *[_record(day, city="Delhi") for day in (0, 1)],
        *[_record(day, city="Chandigarh") for day in (0, 1, 2)],
        _record(3, city="Delhi"),
    ]
    region_result = evaluate_expanding_window(
        region_records,
        minimum_training_dates=3,
        existing_map_path=None,
    )
    region_test = next(case for case in region_result["records"] if case["test_date"] == "2026-09-04")
    assert region_test["skill_source"] == "region"
    assert region_test["training_sample_count"] == {"gfs": 5, "gefs": 5}

    india_records = [
        *[_record(day, city="Mumbai") for day in (0, 1)],
        *[_record(day, city="Delhi") for day in (0, 1, 2)],
        _record(3, city="Mumbai"),
    ]
    india_result = evaluate_expanding_window(
        india_records,
        minimum_training_dates=3,
        existing_map_path=None,
    )
    india_test = next(case for case in india_result["records"] if case["test_date"] == "2026-09-04")
    assert india_test["skill_source"] == "india"
    assert india_test["training_sample_count"] == {"gfs": 5, "gefs": 5}


def test_temporal_report_includes_improvements_and_win_rates():
    records = [_record(day, gfs=10, gefs=14, obs=9) for day in range(6)]
    report = evaluate_expanding_window(records, minimum_training_dates=3, existing_map_path=None)["report"]
    overall = report["metrics"]["overall"]
    assert overall["gfs"]["sample_count"] == 3
    assert overall["win_rates"]["gfs"]["cases"] == 3
    assert overall["improvement_percent"]["gfs"] < 0
    assert overall["win_rates"]["gfs"]["percent"] == 0
    assert overall["improvement_percent"]["equal_weight"] > 0
    assert overall["win_rates"]["equal_weight"]["percent"] == 100
    assert report["provenance"]["observation_source"] == "Open-Meteo Archive API"
    assert report["provenance"]["test_dates_used"] == [
        "2026-09-04", "2026-09-05", "2026-09-06",
    ]


def test_insufficient_training_date_threshold_cannot_be_lowered():
    with pytest.raises(ValueError, match="existing hierarchy threshold"):
        evaluate_expanding_window([_record(0)], minimum_training_dates=1, existing_map_path=None)


def test_existing_evaluation_map_wrapper_is_summarized(tmp_path):
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"metadata": {}, "weight_map": [{"skill_source": "city"}]}))
    records = [_record(day) for day in range(4)]
    result = evaluate_expanding_window(records, existing_map_path=map_path)
    assert result["report"]["existing_1820_cell_validation_map"] == {
        "grid_cells": 1,
        "city_skill_cells": 1,
        "region_skill_cells": 0,
        "india_skill_cells": 0,
    }
