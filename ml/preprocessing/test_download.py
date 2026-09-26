"""Unit tests for the mocked NOMADS subset downloader."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.preprocessing.download import GFS_SUBSET_SEARCHES, download_gfs_subsets


def test_download_gfs_subsets_uses_verified_searches_and_naive_utc(monkeypatch, tmp_path):
    calls = []

    class FakeHerbie:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def download(self, *, search, save_dir, source, overwrite, errors):
            calls.append(("download", search, Path(save_dir)))
            path = Path(save_dir) / f"{len([call for call in calls if call[0] == 'download'])}.grib2"
            path.touch()
            return path

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    date = datetime(2026, 9, 22, 11, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    paths = download_gfs_subsets(date, 24, save_dir=tmp_path)

    init_kwargs = calls[0][1]
    assert init_kwargs["date"] == datetime(2026, 9, 22, 6)
    assert init_kwargs["date"].tzinfo is None
    assert init_kwargs["model"] == "gfs"
    assert init_kwargs["product"] == "pgrb2.0p25"
    assert init_kwargs["priority"] == ["nomads"]
    assert [call[1] for call in calls[1:]] == list(GFS_SUBSET_SEARCHES.values())
    assert set(paths) == set(GFS_SUBSET_SEARCHES)
    assert all(path.is_file() for path in paths.values())
    assert paths.subset_sources == {name: "nomads" for name in GFS_SUBSET_SEARCHES}


def test_download_gfs_subsets_reports_missing_nomads_file(monkeypatch, tmp_path):
    class FakeHerbie:
        def __init__(self, **kwargs):
            pass

        def download(self, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))

    with pytest.raises(RuntimeError, match="Could not download any GFS subset"):
        download_gfs_subsets(datetime(2026, 9, 22, 6), 24, save_dir=tmp_path)


def test_download_gfs_subsets_falls_back_after_source_timeout(monkeypatch, tmp_path):
    attempts = []

    class FakeHerbie:
        def __init__(self, **kwargs):
            self.source = kwargs["priority"][0]

        def download(self, *, search, save_dir, source, overwrite, errors):
            attempts.append((self.source, source, search))
            if source == "nomads":
                raise TimeoutError("timed out")
            path = Path(save_dir) / f"{len(attempts)}.grib2"
            path.touch()
            return path

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    result = download_gfs_subsets(datetime(2026, 9, 22, 6), 24, save_dir=tmp_path)

    assert set(result) == set(GFS_SUBSET_SEARCHES)
    assert {attempt[0] for attempt in attempts} == {"nomads", "aws"}
    assert all(attempt[1] == attempt[0] for attempt in attempts)
    assert len(result.source_failures) == len(GFS_SUBSET_SEARCHES)


def test_download_gfs_subsets_keeps_unrelated_subsets_when_one_fails(monkeypatch, tmp_path):
    class FakeHerbie:
        def __init__(self, **kwargs):
            self.source = kwargs["priority"][0]

        def download(self, *, search, save_dir, source, overwrite, errors):
            if search == GFS_SUBSET_SEARCHES["temperature"]:
                raise TimeoutError("temperature timed out")
            path = Path(save_dir) / f"{source}-{len(list(Path(save_dir).glob('*.grib2')))}.grib2"
            path.touch()
            return path

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    result = download_gfs_subsets(datetime(2026, 9, 22, 6), 24, save_dir=tmp_path)

    assert "temperature" not in result
    assert set(result) == {"wind_u", "wind_v", "precipitation"}
    assert result.missing_subsets == ["temperature"]