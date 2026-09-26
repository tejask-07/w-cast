"""Tests for extreme-event confusion-matrix validation."""

from ml.evaluation.extreme_event_validation import (
    compute_confusion_matrix,
    event_flag,
    summarize_event_metrics,
    validate_extreme_events,
)


def test_event_thresholds_match_detector():
    assert event_flag(49.9, "heavy_rain") is False
    assert event_flag(50.0, "heavy_rain") is True
    assert event_flag(39.9, "heat_wave") is False
    assert event_flag(40.0, "heat_wave") is True
    assert event_flag(16.9, "high_wind") is False
    assert event_flag(17.0, "high_wind") is True


def test_confusion_matrix_and_metrics_are_safe_with_zero_positive_cases():
    actual = [False, False, False]
    predicted = [False, False, False]

    matrix = compute_confusion_matrix(actual, predicted)
    assert matrix["tp"] == 0
    assert matrix["fp"] == 0
    assert matrix["fn"] == 0
    assert matrix["tn"] == 3
    assert matrix["precision"] is None
    assert matrix["recall"] is None
    assert matrix["f1"] is None
    assert matrix["accuracy"] == 1.0
    assert matrix["false_alarm_rate"] == 0.0
    assert matrix["warning"]


def test_validation_summary_handles_small_samples_without_crashing():
    records = [
        {
            "city": "Mumbai",
            "lead_hours": 24,
            "variable": "precipitation",
            "gfs": 0.0,
            "gefs": 0.0,
            "observation": 10.0,
        },
        {
            "city": "Delhi",
            "lead_hours": 48,
            "variable": "temperature",
            "gfs": 42.0,
            "gefs": 39.0,
            "observation": 41.0,
        },
    ]

    summary = validate_extreme_events(records)
    assert "groups" in summary
    assert "warnings" in summary
    assert summary["groups"]["heavy_rain"]["gfs"]["sample_count"] >= 0


def test_summarize_event_metrics_rejects_mismatched_lengths():
    try:
        summarize_event_metrics([True, False], [True])
        raise AssertionError("summarize_event_metrics should reject mismatched lengths")
    except ValueError:
        pass
