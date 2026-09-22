"""Historical observations from the Open-Meteo archive API."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = "temperature_2m,precipitation,wind_speed_10m"


def _as_date(value: str | date | datetime, name: str) -> date:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date or date-like value") from exc


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    latitude_value = float(latitude)
    longitude_value = float(longitude)
    if not -90 <= latitude_value <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= longitude_value <= 180:
        raise ValueError("longitude must be between -180 and 180")
    return latitude_value, longitude_value


def _parse_utc_timestamp(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_response(payload: dict[str, Any], latitude: float, longitude: float, start: date, end: date) -> dict[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError("Open-Meteo response is missing the hourly object")
    required = ("time", "temperature_2m", "precipitation", "wind_speed_10m")
    missing = [name for name in required if name not in hourly]
    if missing:
        raise RuntimeError(f"Open-Meteo response is missing required variables: {missing}")
    timestamps = hourly["time"]
    if not all(len(hourly[name]) == len(timestamps) for name in required[1:]):
        raise RuntimeError("Open-Meteo hourly variables do not have matching timestamps")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
        "hourly": [_parse_utc_timestamp(value) for value in timestamps],
        "temperature_C": list(hourly["temperature_2m"]),
        "precipitation_mm": list(hourly["precipitation"]),
        "wind_speed_ms": list(hourly["wind_speed_10m"]),
    }


def get_historical_observations(
    latitude: float,
    longitude: float,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> dict[str, Any]:
    """Fetch hourly UTC observations for a coordinate and inclusive date range."""
    latitude_value, longitude_value = _validate_coordinates(latitude, longitude)
    start = _as_date(start_date, "start_date")
    end = _as_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    params = {
        "latitude": latitude_value,
        "longitude": longitude_value,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARIABLES,
        "timezone": "UTC",
    }
    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo historical observations request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo returned invalid JSON: {exc}") from exc
    return _parse_response(payload, latitude_value, longitude_value, start, end)


def get_city_historical_observations(city: str, start_date: str | date | datetime, end_date: str | date | datetime) -> dict[str, Any]:
    """Fetch historical observations for one of the MVP cities."""
    from ml.preprocessing.align import CITIES

    try:
        latitude, longitude = CITIES[city]
    except KeyError as exc:
        raise ValueError(f"Unknown city '{city}'. Available cities: {sorted(CITIES)}") from exc
    return get_historical_observations(latitude, longitude, start_date, end_date)


def get_observation_at_time(latitude: float, longitude: float, valid_time: str | datetime) -> dict[str, Any]:
    """Return the nearest hourly observation to a requested UTC timestamp."""
    requested = _parse_utc_timestamp(valid_time)
    observations = get_historical_observations(latitude, longitude, requested.date(), requested.date())
    if not observations["hourly"]:
        raise RuntimeError("Open-Meteo returned no hourly timestamps")
    nearest_index = min(range(len(observations["hourly"])), key=lambda index: abs(observations["hourly"][index] - requested))
    return {
        "valid_time": observations["hourly"][nearest_index],
        "temperature_C": observations["temperature_C"][nearest_index],
        "precipitation_mm": observations["precipitation_mm"][nearest_index],
        "wind_speed_ms": observations["wind_speed_ms"][nearest_index],
    }