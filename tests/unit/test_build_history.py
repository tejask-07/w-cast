"""Offline tests for historical record construction."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from ml.evaluation import build_history


INIT = datetime(2026, 9, 20, 6, tzinfo=timezone.utc)


def _forecast(value):
    return {"temperature_C": value, "precipitation_mm": value / 10, "wind_speed_ms": value / 2}


def _observations(*times):
    return {
        "hourly": list(times),
        "temperature_C": [30.0 + index for index in range(len(times))],
        "precipitation_mm": [index + 1.0 for index in range(len(times))],
        "wind_speed_ms": [4.0 + index for index in range(len(times))],
    }


@pytest.fixture
def offline_sources(monkeypatch, tmp_path):
    calls = {"gfs": [], "gefs": [], "obs": []}

    def download_gfs(date, lead):
        calls["gfs"].append((date, lead))
        return {"source": tmp_path / "gfs.grib2"}

    def download_gefs(date, lead):
        calls["gefs"].append((date, lead))
        return {"source": tmp_path / "gefs.grib2"}

    monkeypatch.setattr(build_history, "download_gfs_subsets", download_gfs)
    monkeypatch.setattr(build_history, "download_gefs_subsets", download_gefs)
    monkeypatch.setattr(build_history, "extract_gfs_point", lambda *_args: _forecast(20.0))
    monkeypatch.setattr(build_history, "extract_gefs_point", lambda *_args: _forecast(21.0))

    def observations(latitude, longitude, start, end):
        calls["obs"].append((latitude, longitude, start, end))
        return _observations(INIT + timedelta(hours=24))

    monkeypatch.setattr(build_history, "get_historical_observations", observations)
    return calls


def test_matches_exact_valid_time_for_each_city_and_variable(offline_sources):
    result = build_history.build_historical_records([INIT], cities=["Mumbai", "Delhi"])
    assert len(result["records"]) == 6
    assert {record["city"] for record in result["records"]} == {"Mumbai", "Delhi"}
    assert {record["variable"] for record in result["records"]} == {"temperature", "precipitation", "wind_speed"}
    record = result["records"][0]
    assert record["valid_time"] == INIT + timedelta(hours=24)
    assert record["gfs"] == 20.0
    assert record["gefs"] == 21.0
    assert record["observation"] == 30.0
    assert {item["variable"]: item["observation"] for item in result["records"]} == {
        "temperature": 30.0,
        "precipitation": 1.0,
        "wind_speed": 4.0,
    }
    assert len(offline_sources["obs"]) == 2


def test_twenty_location_dataset_reuses_forecasts_and_adds_regions(offline_sources):
    result = build_history.build_historical_records(
        [INIT],
        locations=20,
        variables=["temperature"],
    )

    assert len(result["records"]) == 20
    assert len(offline_sources["gfs"]) == 1
    assert len(offline_sources["gefs"]) == 1
    assert len(offline_sources["obs"]) == 20
    assert {record["city"] for record in result["records"]} == set(build_history.INDIA_LOCATIONS)
    assert {record["region"] for record in result["records"]} == {
        "west_coast",
        "north",
        "east",
        "southeast_coast",
        "west",
        "northwest",
        "himalayan",
        "central",
        "east_coast",
        "northeast",
        "south",
        "south_central",
        "southwest_coast",
    }


def test_duplicate_downloads_are_avoided_and_utc_is_normalized(offline_sources):
    naive = INIT.replace(tzinfo=None)
    result = build_history.build_historical_records([naive, INIT, INIT.isoformat().replace("+00:00", "Z")], cities=["Mumbai"], variables=["temperature"])
    assert len(result["records"]) == 1
    assert len(offline_sources["gfs"]) == len(offline_sources["gefs"]) == 1
    assert result["records"][0]["forecast_init_time"].tzinfo == timezone.utc


@pytest.mark.parametrize("missing", ["gfs", "gefs"])
def test_missing_forecast_is_preserved(offline_sources, monkeypatch, missing):
    monkeypatch.setattr(build_history, f"download_{missing}_subsets", lambda *_args: (_ for _ in ()).throw(FileNotFoundError("missing")))
    record = build_history.build_historical_records([INIT], cities=["Mumbai"], variables=["temperature"])["records"][0]
    assert record[missing] is None


def test_missing_observation_is_skipped_and_reason_recorded(offline_sources, monkeypatch):
    monkeypatch.setattr(build_history, "get_historical_observations", lambda *_args: _observations(INIT + timedelta(hours=23)))
    result = build_history.build_historical_records([INIT], cities=["Mumbai"], variables=["temperature"])
    assert result["records"] == []
    assert "exact valid_time" in result["skipped"][0]["reason"]


def test_json_save_and_load(offline_sources, tmp_path):
    path = tmp_path / "history.json"
    result = build_history.build_historical_records([INIT], cities=["Mumbai"], variables=["temperature"], output_path=path)
    loaded = build_history.load_historical_records(path)
    assert loaded == result
    assert path.read_text(encoding="utf-8").count("Z") >= 2


def test_resume_skips_completed_combinations_and_deduplicates(offline_sources, tmp_path):
    path = tmp_path / "history.json"
    first = build_history.build_historical_records(
        [INIT],
        cities=["Mumbai"],
        output_path=path,
    )
    first_downloads = {
        source: len(offline_sources[source])
        for source in ("gfs", "gefs")
    }
    first_observations = len(offline_sources["obs"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"].append(payload["records"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = build_history.build_historical_records(
        [INIT],
        cities=["Mumbai"],
        output_path=path,
        resume=True,
    )

    assert len(first["records"]) == 3
    assert len(resumed["records"]) == 3
    assert len(offline_sources["gfs"]) == first_downloads["gfs"]
    assert len(offline_sources["gefs"]) == first_downloads["gefs"]
    assert len(offline_sources["obs"]) == first_observations


def test_future_dates_are_rejected(offline_sources):
    with pytest.raises(ValueError, match="future"):
        build_history.build_historical_records([datetime.now(timezone.utc) + timedelta(days=2)])