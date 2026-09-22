"""Matched forecast-versus-observation verification for historical skill."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping

import pandas as pd

from ml.evaluation.metrics import bias, mae, rmse
from ml.regimes.classifier import classify_regime

VARIABLE_FIELDS = {
    "temperature": "temperature_C",
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}


def _utc_datetime(value: datetime | str, name: str) -> datetime:
    """Parse a timestamp and normalize naive values as UTC."""
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid datetime") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or None") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite when present")
    return numeric


def _forecast_value(forecasts: Mapping[str, Any], model: str, field: str) -> float | None:
    value = forecasts.get(model)
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get(field)
    return _finite_or_none(value, f"{model}.{field}")


def _observation_value(observation: Mapping[str, Any], field: str) -> float | None:
    return _finite_or_none(observation.get(field), f"observation.{field}")


def build_verification_record(
    forecast_init_time: datetime | str,
    valid_time: datetime | str,
    city: str,
    lead_hours: int,
    variable: str,
    forecasts: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one matched forecast/observation record for GFS and GEFS."""
    from ml.preprocessing.align import CITIES

    if city not in CITIES:
        raise ValueError(f"Unknown city '{city}'. Available cities: {sorted(CITIES)}")
    if variable not in VARIABLE_FIELDS:
        raise ValueError(f"Unsupported variable '{variable}'. Supported: {sorted(VARIABLE_FIELDS)}")
    if not isinstance(lead_hours, int) or lead_hours < 0:
        raise ValueError("lead_hours must be a non-negative integer")
    init = _utc_datetime(forecast_init_time, "forecast_init_time")
    valid = _utc_datetime(valid_time, "valid_time")
    if init + pd.Timedelta(hours=lead_hours).to_pytimedelta() != valid:
        raise ValueError("forecast_init_time plus lead_hours must equal valid_time")
    field = VARIABLE_FIELDS[variable]
    observed = _observation_value(observation, field)
    if observed is None:
        raise ValueError(f"observation.{field} must be finite")
    latitude, longitude = CITIES[city]
    gfs = _forecast_value(forecasts, "gfs", field)
    gefs = _forecast_value(forecasts, "gefs", field)
    return {
        "forecast_init_time": init,
        "valid_time": valid,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "lead_hours": lead_hours,
        "variable": variable,
        "gfs": gfs,
        "gefs": gefs,
        "observation": observed,
        "gfs_error": None if gfs is None else gfs - observed,
        "gefs_error": None if gefs is None else gefs - observed,
        "gfs_absolute_error": None if gfs is None else abs(gfs - observed),
        "gefs_absolute_error": None if gefs is None else abs(gefs - observed),
    }


def _skill_rows(records: list[Mapping[str, Any]], contextual: bool = False) -> pd.DataFrame:
    groups: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for record in records:
        group = (record["model"], record["variable"], record.get("city"), record.get("lead_hours"), record.get("regime")) if contextual else (record["model"], record["variable"])
        if record.get("forecast") is not None and record.get("observation") is not None:
            groups.setdefault(group, []).append((record["observation"], record["forecast"]))
    rows = []
    for group, pairs in groups.items():
        actual, predicted = zip(*pairs)
        row = {"model": group[0], "variable": group[1], "sample_count": len(pairs), "mae": mae(actual, predicted), "rmse": rmse(actual, predicted), "bias": bias(actual, predicted)}
        if contextual:
            row.update({"city": group[2], "lead_hours": group[3], "regime": group[4]})
        rows.append(row)
    columns = ["model", "variable", "sample_count", "mae", "rmse", "bias"]
    if contextual:
        columns += ["city", "lead_hours", "regime"]
    return pd.DataFrame(rows, columns=columns)


def summarize_historical_skill(records: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize MAE, RMSE, and bias by model and variable."""
    normalized = [{"model": model, "variable": record["variable"], "forecast": record.get(model), "observation": record.get("observation")}
                  for record in records for model in ("gfs", "gefs")]
    return _skill_rows(normalized)


def calculate_contextual_skill(records: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize skill by model, variable, city, lead time, and regime."""
    normalized = []
    for record in records:
        precipitation = record.get("precipitation_mm")
        if precipitation is None and record["variable"] == "precipitation":
            precipitation = record.get("observation")
        regime = "UNKNOWN"
        if precipitation is not None:
            try:
                if isfinite(float(precipitation)) and float(precipitation) >= 0:
                    regime = classify_regime(float(precipitation))
            except (TypeError, ValueError):
                pass
        for model in ("gfs", "gefs"):
            normalized.append({"model": model, "variable": record["variable"], "forecast": record.get(model), "observation": record.get("observation"), "city": record.get("city"), "lead_hours": record.get("lead_hours"), "regime": regime})
    return _skill_rows(normalized, contextual=True)