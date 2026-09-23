from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.services.gfs_forecast_service import (
    map_api_variable_to_gfs,
    select_gfs_cycle,
)


client = TestClient(app)


def test_select_gfs_cycle_uses_latest_completed_utc_cycle():
    now = datetime(2026, 9, 22, 11, 30, tzinfo=timezone.utc)

    assert select_gfs_cycle(now) == datetime(
        2026,
        9,
        22,
        6,
        0,
        tzinfo=timezone.utc,
    )


def test_map_api_variable_to_gfs_fields():
    assert map_api_variable_to_gfs("temperature") == "temperature_C"
    assert map_api_variable_to_gfs("rainfall") == "precipitation_mm"
    assert map_api_variable_to_gfs("wind_speed") == "wind_speed_ms"


def test_gfs_adapter_uses_mocked_subset_data(monkeypatch):
    from app.services import gfs_forecast_service

    called = {}

    def fake_download(date, forecast_hour, save_dir):
        called["date"] = date
        called["forecast_hour"] = forecast_hour
        called["save_dir"] = save_dir

        return {
            "temperature": "t.grib2",
            "wind_u": "u.grib2",
            "wind_v": "v.grib2",
            "precipitation": "p.grib2",
        }

    def fake_extract(file_path, lat, lon, lead_hours):
        assert file_path == {
            "temperature": "t.grib2",
            "wind_u": "u.grib2",
            "wind_v": "v.grib2",
            "precipitation": "p.grib2",
        }

        assert lat == 19.07
        assert lon == 72.87
        assert lead_hours == 24

        return {
            "temperature_C": 26.5,
            "precipitation_mm": 12.8,
            "wind_speed_ms": 18.7,
        }

    monkeypatch.setattr(
        gfs_forecast_service,
        "download_gfs_subsets",
        fake_download,
    )

    monkeypatch.setattr(
        gfs_forecast_service,
        "extract_gfs_point",
        fake_extract,
    )

    result = gfs_forecast_service.generate_real_gfs_forecast(
        19.07,
        72.87,
        24,
        "rainfall",
    )

    assert result == {
        "temperature": 26.5,
        "rainfall": 12.8,
        "wind_speed": 18.7,
    }

    assert called["forecast_hour"] == 24


def test_gfs_adapter_failure_becomes_service_error(monkeypatch):
    from app.services import gfs_forecast_service

    def fake_download(*args, **kwargs):
        raise RuntimeError("NOMADS unavailable")

    monkeypatch.setattr(
        gfs_forecast_service,
        "download_gfs_subsets",
        fake_download,
    )

    try:
        gfs_forecast_service.generate_real_gfs_forecast(
            19.07,
            72.87,
            24,
            "rainfall",
        )

        assert False, "Expected RuntimeError from GFS data failure"

    except RuntimeError as exc:
        assert "GFS data unavailable" in str(exc)


def test_health_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_valid_request(monkeypatch):
    from app.services import forecast_service

    def fake_generate_forecast_ml(**kwargs):
        return {
            "forecast": {
                "temperature": 26.5,
                "precipitation": 12.8,
                "wind_speed": 18.7,
            },
            "all_weights": {
                "temperature": {
                    "gfs": 0.4,
                    "gefs": 0.6,
                },
                "precipitation": {
                    "gfs": 0.3,
                    "gefs": 0.7,
                },
                "wind_speed": {
                    "gfs": 0.5,
                    "gefs": 0.5,
                },
            },
            "regime": "NORMAL",
            "extremes": {
                "heavy_rain": False,
                "heat_wave": False,
                "high_wind": True,
            },
        }

    monkeypatch.setattr(
        forecast_service,
        "generate_forecast_ml",
        fake_generate_forecast_ml,
    )

    monkeypatch.setattr(
        forecast_service,
        "_download_forecast_sources",
        lambda lead_hours: ({}, {}),
    )

    response = client.get(
        "/api/forecast",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 24,
            "variable": "rainfall",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["location"]["name"] == "Mumbai"
    assert data["lead_hours"] == 24
    assert "forecast" in data
    assert data["forecast"]["rainfall"] == 12.8
    assert "weights" in data
    assert "regime" in data
    assert "extremes" in data


def test_invalid_latitude_rejected():
    response = client.get(
        "/api/forecast",
        params={
            "lat": -91,
            "lon": 72.87,
            "lead_hours": 24,
            "variable": "rainfall",
        },
    )

    assert response.status_code in {400, 422}


def test_invalid_longitude_rejected():
    response = client.get(
        "/api/forecast",
        params={
            "lat": 19.07,
            "lon": 181,
            "lead_hours": 24,
            "variable": "rainfall",
        },
    )

    assert response.status_code in {400, 422}


def test_invalid_lead_hours_rejected():
    response = client.get(
        "/api/forecast",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 0,
            "variable": "rainfall",
        },
    )

    assert response.status_code in {400, 422}


def test_unsupported_variable_rejected():
    response = client.get(
        "/api/forecast",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 24,
            "variable": "humidity",
        },
    )

    assert response.status_code in {400, 422}


def test_models_endpoint_returns_expected_list():
    response = client.get("/api/models")

    assert response.status_code == 200

    data = response.json()

    assert "models" in data

    ids = [item["id"] for item in data["models"]]

    assert "gfs" in ids
    assert "gefs" in ids
    assert "baseline" in ids


def test_weights_endpoint_returns_expected_structure():
    response = client.get(
        "/api/weights",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 24,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["location"]["lat"] == 19.07
    assert data["location"]["lon"] == 72.87
    assert data["lead_hours"] == 24

    assert set(data["weights"]) == {
        "gfs",
        "gefs",
        "baseline",
    }

    assert data["regime"] in {
        "DRY",
        "NORMAL",
        "WET",
        "EXTREME",
    }


def test_extremes_endpoint_returns_expected_structure():
    response = client.get(
        "/api/extremes",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 24,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "heavy_rain" in data
    assert "heat_wave" in data
    assert "high_wind" in data
    assert "risk_level" in data


def test_verification_endpoint_returns_expected_metrics():
    response = client.get(
        "/api/verification",
        params={
            "lat": 19.07,
            "lon": 72.87,
            "lead_hours": 24,
            "variable": "rainfall",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["variable"] == "rainfall"
    assert data["lead_hours"] == 24

    assert "gfs" in data["metrics"]
    assert "gefs" in data["metrics"]
    assert "adaptive_blend" in data["metrics"]

    assert set(data["metrics"]["gfs"]) == {
        "mae",
        "rmse",
        "bias",
    }