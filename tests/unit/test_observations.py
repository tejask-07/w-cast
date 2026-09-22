"""Offline tests for Open-Meteo historical observations."""

from datetime import datetime, timezone

import pytest

from ml.preprocessing import observations


def _response(payload, status_code=200):
    class FakeResponse:
        def raise_for_status(self):
            if status_code >= 400:
                import requests

                raise requests.HTTPError(f"status={status_code}")

        def json(self):
            return payload

    return FakeResponse()


PAYLOAD = {
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        "temperature_2m": [20.0, 20.5, 21.0],
        "precipitation": [0.0, 1.2, 0.4],
        "wind_speed_10m": [3.0, 4.0, 5.0],
    }
}


def test_successful_historical_response_parsing(monkeypatch):
    monkeypatch.setattr(observations.requests, "get", lambda *args, **kwargs: _response(PAYLOAD))
    result = observations.get_historical_observations(19.076, 72.8777, "2024-01-01", "2024-01-01")
    assert result["timezone"] == "UTC"
    assert result["hourly"][0].tzinfo == timezone.utc
    assert result["temperature_C"] == [20.0, 20.5, 21.0]


def test_city_lookup(monkeypatch):
    captured = {}

    def fake_fetch(latitude, longitude, start_date, end_date):
        captured.update(latitude=latitude, longitude=longitude)
        return {"latitude": latitude, "longitude": longitude}

    monkeypatch.setattr(observations, "get_historical_observations", fake_fetch)
    result = observations.get_city_historical_observations("Mumbai", "2024-01-01", "2024-01-01")
    assert result["latitude"] == 19.076
    assert captured["longitude"] == 72.8777


def test_invalid_city():
    with pytest.raises(ValueError, match="Unknown city"):
        observations.get_city_historical_observations("Pune", "2024-01-01", "2024-01-01")


def test_invalid_date_range():
    with pytest.raises(ValueError, match="on or before"):
        observations.get_historical_observations(0, 0, "2024-01-02", "2024-01-01")


def test_api_failure(monkeypatch):
    monkeypatch.setattr(observations.requests, "get", lambda *args, **kwargs: _response({}, 500))
    with pytest.raises(RuntimeError, match="request failed"):
        observations.get_historical_observations(0, 0, "2024-01-01", "2024-01-01")


def test_missing_required_variables(monkeypatch):
    monkeypatch.setattr(observations.requests, "get", lambda *args, **kwargs: _response({"hourly": {"time": []}}))
    with pytest.raises(RuntimeError, match="missing required variables"):
        observations.get_historical_observations(0, 0, "2024-01-01", "2024-01-01")


def test_nearest_valid_time_selection(monkeypatch):
    monkeypatch.setattr(observations.requests, "get", lambda *args, **kwargs: _response(PAYLOAD))
    result = observations.get_observation_at_time(19.076, 72.8777, datetime(2024, 1, 1, 1, 20, tzinfo=timezone.utc))
    assert result == {
        "valid_time": datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        "temperature_C": 20.5,
        "precipitation_mm": 1.2,
        "wind_speed_ms": 4.0,
    }