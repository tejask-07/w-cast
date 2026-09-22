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

        def download(self, *, search, save_dir):
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


def test_download_gfs_subsets_reports_missing_nomads_file(monkeypatch, tmp_path):
    class FakeHerbie:
        def __init__(self, **kwargs):
            pass

        def download(self, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))

    with pytest.raises(RuntimeError, match="returned no file"):
        download_gfs_subsets(datetime(2026, 9, 22, 6), 24, save_dir=tmp_path)