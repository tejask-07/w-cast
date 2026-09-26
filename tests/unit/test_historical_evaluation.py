import json
import math

import pytest

from ml.evaluation.evaluate_paired_dataset import (
    _improvement,
    _metric_summary,
    evaluate_records,
    run_evaluation,
)
from ml.evaluation.historical_skill import select_hierarchical_skill


def _records():
    return [
        {
            "location_name": "Mumbai",
            "latitude": 19.076,
            "longitude": 72.8777,
            "region": "west_coast",
            "forecast_date": f"2026-09-{12 + index}",
            "forecast_cycle": f"2026-09-{12 + index}T06:00:00Z",
            "valid_time": f"2026-09-{13 + index}T06:00:00Z",
            "variable": "temperature",
            "lead_hours": 24,
            "gfs_value": 11.0,
            "gefs_value": 13.0,
            "observation_value": 10.0,
            "gfs_error": 1.0,
            "gefs_error": 3.0,
            "gfs_abs_error": 1.0,
            "gefs_abs_error": 3.0,
            "gfs_source": "GFS fixture source",
            "gfs_resolution_degrees": 0.25,
            "gefs_source": "GEFS operational archive (NOAA AWS Open Data)",
            "gefs_resolution_degrees": 0.25,
        }
        for index in range(3)
    ]


def test_mae_rmse_and_bias_metrics():
    assert _metric_summary([10.0, 10.0, 10.0], [11.0, 12.0, 13.0]) == {
        "mae": 2.0,
        "rmse": math.sqrt(14 / 3),
        "bias": 2.0,
        "sample_count": 3,
    }


def test_equal_weight_and_wcast_metrics_and_improvements():
    report, _ = evaluate_records(_records())
    overall = report["metrics"]["overall"]
    assert overall["gfs"]["mae"] == 1.0
    assert overall["gefs"]["mae"] == 3.0
    assert overall["equal_weight"]["mae"] == 2.0
    assert overall["wcast"]["mae"] == pytest.approx(1.5, abs=1e-6)
    assert overall["wcast"]["rmse"] == pytest.approx(1.5, abs=1e-6)
    assert overall["wcast"]["bias"] == pytest.approx(1.5, abs=1e-6)
    assert overall["improvement_vs_gfs_percent"] == pytest.approx(-50.0, abs=1e-4)
    assert overall["improvement_vs_gefs_percent"] == pytest.approx(50.0, abs=1e-4)
    assert overall["improvement_vs_equal_percent"] == pytest.approx(25.0, abs=1e-4)


def test_inverse_mae_weights_are_normalized_and_city_threshold_is_used():
    report, _ = evaluate_records(_records())
    weight = report["location_weight_resolutions"][0]
    assert weight["skill_source"] == "city"
    assert weight["sample_count"] == 3
    assert math.isclose(weight["gfs_weight"], 0.75, abs_tol=1e-6)
    assert math.isclose(weight["gefs_weight"], 0.25, abs_tol=1e-6)
    assert math.isclose(weight["gfs_weight"] + weight["gefs_weight"], 1.0)


def _skill(sample_count, gfs_mae, gefs_mae):
    return {
        "gfs": {"mae": gfs_mae, "rmse": gfs_mae, "bias": 0.0, "sample_count": sample_count},
        "gefs": {"mae": gefs_mae, "rmse": gefs_mae, "bias": 0.0, "sample_count": sample_count},
    }


def test_hierarchy_falls_back_city_to_region_then_india_at_existing_threshold():
    aggregates = {
        "city": {"Mumbai": {"temperature": {"24": _skill(2, 1.0, 2.0)}}},
        "region": {"west_coast": {"temperature": {"24": _skill(3, 1.5, 2.5)}}},
        "india": {"India": {"temperature": {"24": _skill(6, 2.0, 3.0)}}},
    }
    selected = select_hierarchical_skill(aggregates, "Mumbai", "temperature", 24)
    assert selected["source"] == "region"
    assert selected["sample_count"] == {"gfs": 3, "gefs": 3}

    aggregates["region"]["west_coast"]["temperature"]["24"] = _skill(2, 1.5, 2.5)
    selected = select_hierarchical_skill(aggregates, "Mumbai", "temperature", 24)
    assert selected["source"] == "india"
    assert selected["sample_count"] == {"gfs": 6, "gefs": 6}


def test_evaluation_weight_map_is_separate_and_contains_metadata(tmp_path):
    dataset = tmp_path / "paired.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in _records()) + "\n", encoding="utf-8")
    weight_map_path = tmp_path / "india_weight_map_validation.json"
    report_path = tmp_path / "evaluation.report.json"
    report, weight_map = run_evaluation(dataset, weight_map_path, report_path)

    assert weight_map_path.is_file()
    assert report_path.is_file()
    assert weight_map["metadata"]["evaluation_only"] is True
    assert weight_map["metadata"]["dataset_records"] == 3
    assert weight_map["weight_map"]
    assert report["skill_resolution_counts"] == {"city": 1}


def test_zero_baseline_mae_improvement_is_not_divided_by_zero():
    assert _improvement(0.0, 1.0) is None
    assert _improvement(2.0, 1.0) == 50.0
    assert _improvement(2.0, 3.0) == -50.0