from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ml.preprocessing import historical_gefs


INIT = datetime(2026, 9, 19, 6, tzinfo=timezone.utc)


def _inventory():
    return pd.DataFrame([
        {
            "variable": "TMP",
            "level": "2 m above ground",
            "forecast_time": "24 hour fcst",
            "search_this": ":TMP:2 m above ground:24 hour fcst:ens mean:",
        },
        {
            "variable": "APCP",
            "level": "surface",
            "forecast_time": "18-24 hour acc fcst",
            "search_this": ":APCP:surface:18-24 hour acc fcst:ens mean:",
        },
        {
            "variable": "UGRD",
            "level": "10 m above ground",
            "forecast_time": "24 hour fcst",
            "search_this": ":UGRD:10 m above ground:24 hour fcst:ens mean:",
        },
        {
            "variable": "VGRD",
            "level": "10 m above ground",
            "forecast_time": "24 hour fcst",
            "search_this": ":VGRD:10 m above ground:24 hour fcst:ens mean:",
        },
    ])


def _fake_herbie(monkeypatch, tmp_path, idx="aws-index"):
    calls = {"init": [], "downloads": []}

    class FakeForecast:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)
            self.idx = idx

        def inventory(self):
            return _inventory()

        def download(self, *, search, save_dir):
            calls["downloads"].append(search)
            path = Path(save_dir) / f"subset-{len(calls['downloads'])}.grib2"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture boundary only")
            return path

    monkeypatch.setitem(__import__("sys").modules, "herbie", SimpleNamespace(Herbie=FakeForecast))
    return calls


def test_historical_gefs_success_and_source_metadata(monkeypatch, tmp_path):
    calls = _fake_herbie(monkeypatch, tmp_path)
    monkeypatch.setattr(
        historical_gefs,
        "extract_gefs_point",
        lambda *_args: {"temperature_C": 23.5},
    )
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)

    result = adapter.get_forecast("2026-09-19", 6, 24, "temperature", 19.076, 72.8777)

    assert result["forecast_value"] == 23.5
    assert result["source"] == "noaa-gefs-operational-aws"
    assert result["gefs_source"] == "GEFS operational archive (NOAA AWS Open Data)"
    assert result["spatial_resolution_degrees"] == 0.25
    assert result["initialization_time"] == INIT
    assert result["valid_time"] == datetime(2026, 9, 20, 6, tzinfo=timezone.utc)
    assert calls["init"][0]["priority"] == ["aws"]
    assert calls["downloads"] == [":TMP:2 m above ground:24 hour fcst:ens mean:"]


def test_historical_gefs_missing_archive_fails_explicitly(monkeypatch, tmp_path):
    _fake_herbie(monkeypatch, tmp_path, idx=None)
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)
    with pytest.raises(historical_gefs.HistoricalGEFSUnavailable, match="index unavailable"):
        adapter.get_forecast("2026-09-19", 6, 24, "temperature", 19.076, 72.8777)


def test_historical_gefs_invalid_variable_is_rejected_before_download(monkeypatch, tmp_path):
    calls = _fake_herbie(monkeypatch, tmp_path)
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)
    with pytest.raises(ValueError, match="Unsupported historical GEFS variable"):
        adapter.get_forecast("2026-09-19", 6, 24, "humidity", 19.076, 72.8777)
    assert calls["init"] == []


def test_pair_timestamp_validator_rejects_mismatched_initialization():
    valid = INIT.replace(hour=6) + __import__("datetime").timedelta(hours=24)
    with pytest.raises(ValueError, match="initialization times do not match"):
        historical_gefs.validate_pair_timestamps(
            INIT,
            INIT.replace(hour=12),
            valid,
            valid,
            24,
        )


def test_historical_gefs_never_converts_missing_values_to_zero(monkeypatch, tmp_path):
    _fake_herbie(monkeypatch, tmp_path)
    monkeypatch.setattr(historical_gefs, "extract_gefs_point", lambda *_args: {"temperature_C": None})
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)
    with pytest.raises(historical_gefs.HistoricalGEFSUnavailable, match="is missing"):
        adapter.get_forecast("2026-09-19", 6, 24, "temperature", 19.076, 72.8777)


def test_historical_gefs_nonfinite_values_are_rejected(monkeypatch, tmp_path):
    _fake_herbie(monkeypatch, tmp_path)
    monkeypatch.setattr(historical_gefs, "extract_gefs_point", lambda *_args: {"temperature_C": float("nan")})
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)
    with pytest.raises(historical_gefs.HistoricalGEFSUnavailable, match="non-finite"):
        adapter.get_forecast("2026-09-19", 6, 24, "temperature", 19.076, 72.8777)


def test_historical_gefs_rainfall_sums_three_hour_intervals(monkeypatch, tmp_path):
    adapter = historical_gefs.HistoricalGEFSAdapter(tmp_path)
    requested_intervals = []
    monkeypatch.setattr(adapter, "_forecast", lambda init, lead: object())

    def field_path(forecast, init, lead, variable, field_name, grib_variable, level):
        requested_intervals.append(lead)
        return tmp_path / f"rain-{lead}.grib2"

    monkeypatch.setattr(adapter, "_field_path", field_path)
    monkeypatch.setattr(
        historical_gefs,
        "extract_gefs_point",
        lambda fields, _lat, _lon, lead: {"precipitation_mm": float(lead // 3)},
    )

    result = adapter.get_forecast("2026-09-19", 6, 24, "rainfall", 19.076, 72.8777)

    assert requested_intervals == list(range(3, 25, 3))
    assert result["forecast_value"] == sum(range(1, 9))
    assert result["accumulation_period_hours"] == 24


def test_pair_timestamp_validator_rejects_valid_time_mismatch():
    with pytest.raises(ValueError, match="valid times do not match"):
        historical_gefs.validate_pair_timestamps(
            INIT,
            INIT,
            datetime(2026, 9, 20, 6, tzinfo=timezone.utc),
            datetime(2026, 9, 20, 12, tzinfo=timezone.utc),
            24,
        )