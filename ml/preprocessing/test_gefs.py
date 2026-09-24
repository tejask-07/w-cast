"""Offline tests for GEFS ensemble-mean ingestion."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ml.preprocessing import gefs


def test_download_gefs_subsets_uses_inventory_and_ensemble_mean(monkeypatch, tmp_path):
    calls = []

    class FakeHerbie:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def inventory(self):
            return pd.DataFrame(
                [
                    {"variable": "TMP", "level": "2 m above ground", "search_this": ":TMP:2 m above ground:24 hour fcst:ens mean:"},
                    {"variable": "UGRD", "level": "10 m above ground", "search_this": ":UGRD:10 m above ground:24 hour fcst:ens mean:"},
                    {"variable": "VGRD", "level": "10 m above ground", "search_this": ":VGRD:10 m above ground:24 hour fcst:ens mean:"},
                    {"variable": "APCP", "level": "surface", "search_this": ":APCP:surface:18-24 hour acc fcst:ens mean:"},
                ]
            )

        def download(self, *, search, save_dir):
            calls.append(("download", search))
            path = Path(save_dir) / f"{len([call for call in calls if call[0] == 'download'])}.grib2"
            path.touch()
            return path

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    paths = gefs.download_gefs_subsets(
        datetime(2026, 9, 22, 11, tzinfo=timezone.utc), 24, save_dir=tmp_path
    )

    init = calls[0][1]
    assert init["model"] == "gefs"
    assert init["product"] == "atmos.25"
    assert init["member"] == "avg"
    assert init["priority"] == ["nomads"]
    assert init["date"].tzinfo is None
    assert set(paths) == {"temperature", "wind_u", "wind_v", "precipitation"}
    assert all(path.is_file() for path in paths.values())
    assert calls[-1][1].startswith(":APCP:surface:18-24 hour acc fcst:")


def test_download_gefs_rejects_non_mean_member(tmp_path):
    with pytest.raises(ValueError, match="member='avg'"):
        gefs.download_gefs_subsets(datetime(2026, 9, 22, 6), 24, tmp_path, member="p01")


def test_download_gefs_returns_none_when_archive_index_is_missing(monkeypatch, tmp_path):
    calls = []

    class FakeHerbie:
        idx = None

        def __init__(self, **kwargs):
            calls.append("init")

        def inventory(self):
            calls.append("inventory")
            raise AssertionError("inventory should not run without an index")

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))

    assert gefs.download_gefs_subsets(
        datetime(2026, 9, 22, 6), 24, tmp_path
    ) is None
    assert calls == ["init"]


def test_extract_gefs_point_uses_common_schema(monkeypatch):
    expected = {
        "temperature_C": 28.0,
        "wind_u_ms": 3.0,
        "wind_v_ms": 4.0,
        "wind_speed_ms": 5.0,
        "precipitation_mm": 11.0,
    }
    monkeypatch.setattr(gefs, "extract_gfs_point", lambda *args: dict(expected))

    result = gefs.extract_gefs_point({"temperature": "t", "wind_u": "u", "wind_v": "v", "precipitation": "p"}, 19, 72, 24)

    assert result["wind_speed_ms"] == 5.0
    assert result["ensemble_member"] == "avg"
    assert result["ensemble_representation"] == "official GEFS ensemble mean"