from pathlib import Path

import pytest

from app.services import hourly_forecast_service as hourly


def _fake_paths(prefix: str, hour: int) -> dict[str, Path]:
    return {
        "temperature": Path(f"{prefix}-temperature-{hour}.grib2"),
        "wind_u": Path(f"{prefix}-wind-u-{hour}.grib2"),
        "wind_v": Path(f"{prefix}-wind-v-{hour}.grib2"),
        "precipitation": Path(f"{prefix}-precipitation-{hour}.grib2"),
    }


def _fake_point(paths, lat, lon, lead_hours):
    path_text = str(next(iter(paths.values())))
    value = float(lead_hours)
    result = {
        "temperature_C": 20.0 + value,
        "precipitation_mm": 1.0 + value,
        "wind_speed_ms": 2.0 + value,
    }
    if "gefs-missing" in path_text:
        result["precipitation_mm"] = None
    return result


def _patch_hourly_dependencies(monkeypatch):
    gfs_hours = []
    gefs_hours = []

    monkeypatch.setattr(
        hourly,
        "_download_gfs_hour",
        lambda hour: (gfs_hours.append(hour) or _fake_paths("gfs", hour)),
    )
    monkeypatch.setattr(
        hourly,
        "_download_gefs_hour",
        lambda hour: (gefs_hours.append(hour) or _fake_paths("gefs", hour)),
    )
    monkeypatch.setattr(hourly, "extract_gfs_point", _fake_point)
    monkeypatch.setattr(hourly, "extract_gefs_point", _fake_point)
    monkeypatch.setattr(
        hourly,
        "_blend_value",
        lambda latitude, longitude, variable, lead_hours, gfs_value, gefs_value: (
            gfs_value + gefs_value
        ) / 2,
    )
    return gfs_hours, gefs_hours


@pytest.mark.parametrize("lead_hours, expected_points", [(24, 25), (48, 49), (72, 73)])
def test_hourly_forecast_returns_requested_number_of_points(
    monkeypatch,
    lead_hours,
    expected_points,
):
    _patch_hourly_dependencies(monkeypatch)
    result = hourly.generate_hourly_forecast(19.07, 72.87, lead_hours, "temperature")
    assert len(result["points"]) == expected_points
    assert [point["hour"] for point in result["points"]] == list(range(lead_hours + 1))


def test_hourly_forecast_downloads_gefs_only_at_three_hour_intervals(monkeypatch):
    gfs_hours, gefs_hours = _patch_hourly_dependencies(monkeypatch)
    hourly.generate_hourly_forecast(19.07, 72.87, 24, "temperature")
    assert gfs_hours == list(range(25))
    assert gefs_hours == list(range(0, 25, 3))


@pytest.mark.parametrize("variable", ["temperature", "wind_speed"])
def test_hourly_forecast_supports_temperature_and_wind(monkeypatch, variable):
    _patch_hourly_dependencies(monkeypatch)
    result = hourly.generate_hourly_forecast(19.07, 72.87, 24, variable)
    assert result["variable"] == variable
    assert len(result["points"]) == 25


def test_hourly_rainfall_falls_back_to_gfs_when_gefs_apcp_is_unavailable(monkeypatch):
    _patch_hourly_dependencies(monkeypatch)
    monkeypatch.setattr(
        hourly,
        "_download_gefs_hour",
        lambda hour: _fake_paths("gefs-missing", hour),
    )
    result = hourly.generate_hourly_forecast(19.07, 72.87, 24, "rainfall")
    assert len(result["points"]) == 25
    assert result["points"][0] == {"hour": 0, "value": 0.0}


def test_hourly_accumulations_difference_and_f00():
    assert hourly._hourly_accumulations(
        {1: 1.5, 2: 2.0, 3: 4.25},
        3,
    ) == {
        0: 0.0,
        1: 1.5,
        2: 0.5,
        3: 2.25,
    }


def test_hourly_request_validation_rejects_unsupported_lead():
    with pytest.raises(ValueError, match="24, 48, or 72"):
        hourly.validate_hourly_request("temperature", 1)