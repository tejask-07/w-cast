"""Confusion-matrix validation for the current rule-based extremes detector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.historical_weights import build_historical_weights

EVENT_VARIABLES = {
    "heavy_rain": "precipitation",
    "heat_wave": "temperature",
    "high_wind": "wind_speed",
}

EVENT_THRESHOLDS = {
    "heavy_rain": {"threshold": 50.0, "variable": "precipitation"},
    "heat_wave": {"threshold": 40.0, "variable": "temperature"},
    "high_wind": {"threshold": 17.0, "variable": "wind_speed"},
}

MODEL_NAMES = ("gfs", "gefs", "wcast")


def event_flag(value: Any, event_name: str) -> bool:
    """Return whether the provided scalar exceeds the rule threshold for an event."""
    if event_name not in EVENT_THRESHOLDS:
        raise ValueError(f"Unsupported event '{event_name}'. Expected one of: {sorted(EVENT_THRESHOLDS)}")
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Event value for '{event_name}' must be numeric or None") from exc
    if not isfinite(numeric):
        raise ValueError(f"Event value for '{event_name}' must be finite")
    return numeric >= float(EVENT_THRESHOLDS[event_name]["threshold"])


def compute_confusion_matrix(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    """Compute a standard confusion matrix and derived metrics.

    The logic is intentionally conservative: when positives are absent, precision,
    recall, and F1 are reported as None and a warning is returned instead of
    fabricating a value.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted sequences must have the same length")

    actual_list = [bool(value) for value in actual]
    predicted_list = [bool(value) for value in predicted]
    total = len(actual_list)
    tp = sum(1 for actual_value, predicted_value in zip(actual_list, predicted_list) if actual_value and predicted_value)
    fp = sum(1 for actual_value, predicted_value in zip(actual_list, predicted_list) if (not actual_value) and predicted_value)
    fn = sum(1 for actual_value, predicted_value in zip(actual_list, predicted_list) if actual_value and (not predicted_value))
    tn = sum(1 for actual_value, predicted_value in zip(actual_list, predicted_list) if (not actual_value) and (not predicted_value))

    observed_positives = tp + fn
    predicted_positives = tp + fp
    precision = tp / predicted_positives if predicted_positives else None
    recall = tp / observed_positives if observed_positives else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = None
    accuracy = (tp + tn) / total if total else None
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else None

    warning = None
    if observed_positives < 3:
        warning = "Insufficient observed positive events for stable statistics; the historical window is too short."
    elif predicted_positives == 0:
        warning = "No forecast positive events were generated for this event; precision is undefined."

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sample_count": total,
        "observed_positives": observed_positives,
        "predicted_positives": predicted_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_alarm_rate": false_alarm_rate,
        "warning": warning,
    }


def summarize_event_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    """Return the confusion-matrix metrics with a compact payload."""
    matrix = compute_confusion_matrix(actual, predicted)
    return {
        "sample_count": matrix["sample_count"],
        "tp": matrix["tp"],
        "fp": matrix["fp"],
        "fn": matrix["fn"],
        "tn": matrix["tn"],
        "precision": matrix["precision"],
        "recall": matrix["recall"],
        "f1": matrix["f1"],
        "accuracy": matrix["accuracy"],
        "false_alarm_rate": matrix["false_alarm_rate"],
        "warning": matrix["warning"],
    }


def _record_wcast_value(record: Mapping[str, Any], history_path: str | Path | None = None) -> float | None:
    """Compute the W-CAST forecast value for a record using historical inverse-MAE weights."""
    if record.get("wcast") is not None:
        return float(record["wcast"])

    gfs_value = record.get("gfs")
    gefs_value = record.get("gefs")
    if gfs_value is None or gefs_value is None:
        return None

    city = record.get("city")
    variable = record.get("variable")
    lead_hours = record.get("lead_hours")
    if city is None or variable is None or lead_hours is None:
        return 0.5 * float(gfs_value) + 0.5 * float(gefs_value)

    default_path = "data/processed/history_audited_gefs_2026-09-20_to_2026-09-24_20locations_multilead.json"
    path = Path(history_path) if history_path is not None else Path(default_path)

    try:
        if not path.exists():
            raise FileNotFoundError(path)
        weights_data = build_historical_weights(str(path))
        weights = weights_data.get(str(city), {}).get(str(variable), {}).get(str(lead_hours), {}).get("weights")
    except Exception:
        weights = None

    if weights is None:
        weights = {"gfs": 0.5, "gefs": 0.5}
    gfs_weight = float(weights.get("gfs", 0.5))
    gefs_weight = float(weights.get("gefs", 0.5))
    return gfs_weight * float(gfs_value) + gefs_weight * float(gefs_value)


def _model_value(record: Mapping[str, Any], model: str, history_path: str | Path | None = None) -> float | None:
    """Return the forecast value for a given model from the historical record."""
    if model == "wcast":
        return _record_wcast_value(record, history_path)
    value = record.get(model)
    if value is None:
        return None
    return float(value)


def _filtered_records_for_event(records: Sequence[Mapping[str, Any]], event_name: str) -> list[Mapping[str, Any]]:
    variable = EVENT_VARIABLES[event_name]
    return [record for record in records if str(record.get("variable", "")).lower() == str(variable).lower()]


def _event_values_for_model(records: Sequence[Mapping[str, Any]], event_name: str, model: str, history_path: str | Path | None = None) -> tuple[list[bool], list[bool], list[dict[str, Any]]]:
    event_records: list[dict[str, Any]] = []
    actual: list[bool] = []
    predicted: list[bool] = []

    for record in _filtered_records_for_event(records, event_name):
        observation = record.get("observation")
        if observation is None:
            continue
        try:
            actual_flag = event_flag(observation, event_name)
        except ValueError:
            continue
        model_value = _model_value(record, model, history_path)
        if model_value is None:
            continue
        try:
            predicted_flag = event_flag(model_value, event_name)
        except ValueError:
            continue
        actual.append(actual_flag)
        predicted.append(predicted_flag)
        event_records.append({
            "city": record.get("city"),
            "lead_hours": record.get("lead_hours"),
            "valid_time": record.get("valid_time"),
            "observation": observation,
            "forecast_value": model_value,
            "observed_event": actual_flag,
            "predicted_event": predicted_flag,
        })

    return actual, predicted, event_records


def validate_extreme_events(
    records: Sequence[Mapping[str, Any]],
    history_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the current event logic across GFS, GEFS, and W-CAST forecasts."""
    event_names = sorted(EVENT_VARIABLES)
    groups: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for event_name in event_names:
        event_groups: dict[str, Any] = {}
        for model in MODEL_NAMES:
            actual_values, predicted_values, event_records = _event_values_for_model(records, event_name, model, history_path)
            matrix = compute_confusion_matrix(actual_values, predicted_values)
            event_groups[model] = {
                "sample_count": len(event_records),
                "tp": matrix["tp"],
                "fp": matrix["fp"],
                "fn": matrix["fn"],
                "tn": matrix["tn"],
                "precision": matrix["precision"],
                "recall": matrix["recall"],
                "f1": matrix["f1"],
                "accuracy": matrix["accuracy"],
                "false_alarm_rate": matrix["false_alarm_rate"],
                "warning": matrix["warning"],
                "records": event_records,
            }
            if matrix["warning"]:
                warnings.append(f"{event_name}:{model}: {matrix['warning']}")
        groups[event_name] = event_groups

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_path": str(history_path) if history_path is not None else None,
        "event_names": event_names,
        "groups": groups,
        "warnings": warnings,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _load_history_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return list(records)


def main() -> None:
    default_history = Path("data/processed/history_audited_gefs_2026-09-20_to_2026-09-24_20locations_multilead.json")
    result = validate_extreme_events(
        _load_history_records(default_history),
        history_path=default_history,
        output_path="data/processed/extreme_event_validation.json",
    )
    print(json.dumps({
        "events": sorted(result["groups"]),
        "warnings": result["warnings"],
    }, indent=2))


if __name__ == "__main__":
    main()
