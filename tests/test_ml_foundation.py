"""Synthetic tests for the W-CAST Person 1 ML foundation."""

import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ml.baselines.equal_weight import equal_weight_blend
from ml.baselines.static_weight import static_weight_blend
from ml.blending.adaptive_weights import adaptive_weights
from ml.blending.blender import blend_forecasts
from ml.evaluation.metrics import bias, mae, rmse
from ml.regimes.classifier import classify_regime


def test_download_gfs_normalizes_aware_datetime(monkeypatch, tmp_path):
    from ml.preprocessing.download import download_gfs

    captured = {}

    class FakeHerbie:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def download(self, save_dir):
            path = tmp_path / "forecast.grib2"
            path.touch()
            return path

    monkeypatch.setitem(sys.modules, "herbie", SimpleNamespace(Herbie=FakeHerbie))
    aware_date = datetime(2026, 9, 22, 11, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    result = download_gfs(aware_date, 24, save_dir=tmp_path)

    assert captured["date"] == datetime(2026, 9, 22, 6)
    assert captured["date"].tzinfo is None
    assert captured["priority"] == ["nomads"]
    assert result == tmp_path / "forecast.grib2"


def test_equal_weighting():
    result = equal_weight_blend({"gfs": [2, 4], "gefs": [4, 8]})
    np.testing.assert_allclose(result, [3, 6])


def test_static_weighting():
    result = static_weight_blend({"gfs": [2, 4], "gefs": [4, 8]}, {"gfs": 0.75, "gefs": 0.25})
    np.testing.assert_allclose(result, [2.5, 5])


def test_adaptive_weighting_prefers_lower_error_and_handles_missing():
    weights = adaptive_weights({"gfs": 1.0, "gefs": 3.0, "optional": None})
    assert weights["gfs"] > weights["gefs"] > weights["optional"]
    assert np.isclose(sum(weights.values()), 1.0)


def test_blender_rejects_invalid_weights_and_shapes():
    with pytest.raises(ValueError, match="sum"):
        blend_forecasts({"gfs": [1], "gefs": [2]}, {"gfs": 0.5, "gefs": 0.4})
    with pytest.raises(ValueError, match="shapes"):
        blend_forecasts({"gfs": [1], "gefs": [2, 3]}, {"gfs": 0.5, "gefs": 0.5})
    with pytest.raises(ValueError, match="non-negative"):
        blend_forecasts({"gfs": [1], "gefs": [2]}, {"gfs": -0.1, "gefs": 1.1})


@pytest.mark.parametrize(
    ("rainfall", "expected"),
    [(0, "DRY"), (10, "NORMAL"), (30, "WET"), (100, "EXTREME")],
)
def test_regime_classification(rainfall, expected):
    assert classify_regime(rainfall) == expected


def test_verification_metrics():
    actual = [1, 2, 3]
    forecast = [2, 2, 5]
    assert mae(actual, forecast) == pytest.approx(1.0)
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(5 / 3))
    assert bias(actual, forecast) == pytest.approx(1.0)


def test_nearest_grid_extraction():
    xr = pytest.importorskip("xarray")
    from ml.preprocessing.align import extract_city_grid_point, extract_nearest_grid_point

    dataset = xr.Dataset(
        {"temperature": (("lat", "lon"), np.arange(9).reshape(3, 3))},
        coords={"lat": [13.0, 19.0, 28.0], "lon": [72.0, 77.0, 88.0]},
    )
    point = extract_nearest_grid_point(dataset, 19.076, 72.8777)
    assert point.temperature.item() == 3
    assert extract_city_grid_point(dataset, "Mumbai").temperature.item() == 3


def test_gfs_subset_extraction_converts_units_and_calculates_wind(monkeypatch, tmp_path):
    from ml.preprocessing import gfs_extract

    class FakeSelected:
        def __init__(self, name, value):
            self.data_vars = {name: value}
            self._name = name
            self._value = value

        def __getitem__(self, name):
            class Scalar:
                def __init__(self, value):
                    self.values = np.asarray(value)

            assert name == self._name
            return Scalar(self._value)

    class FakeDataset:
        def __init__(self, name, value):
            self.coords = {"latitude": np.array([19.0, 20.0]), "longitude": np.array([287.0, 288.0])}
            self.variables = self.coords
            self.dims = {"latitude": 2, "longitude": 2}
            self.data_vars = {name: value}
            self.name = name
            self.value = value
            self.selected = None

        def __getitem__(self, name):
            return self.coords[name]

        def sel(self, selection, method):
            self.selected = selection
            return FakeSelected(self.name, self.value)

        def close(self):
            pass

    values = {"temperature": ("t2m", 300.0), "wind_u": ("u10", 3.0), "wind_v": ("v10", 4.0), "precipitation": ("tp", 11.8125)}
    opened = {}

    def fake_open(path):
        key = Path(path).stem
        dataset = FakeDataset(*values[key])
        opened[key] = dataset
        return dataset

    monkeypatch.setattr(gfs_extract, "open_gfs_subset", fake_open)
    paths = {key: tmp_path / f"{key}.grib2" for key in values}
    for path in paths.values():
        path.touch()
    result = gfs_extract.extract_gfs_point(paths, 19.076, -72.8777, lead_hours=24)

    assert result["temperature_C"] == pytest.approx(26.85)
    assert result["wind_speed_ms"] == pytest.approx(5.0)
    assert result["precipitation_mm"] == pytest.approx(11.8125)
    assert result["temperature_source"] == "GFS TMP 2m"
    assert result["wind_source"] == "GFS UGRD/VGRD 10m"
    assert opened["temperature"].selected["longitude"] == pytest.approx(287.1223)


def test_gfs_subset_missing_field_is_explicit(monkeypatch, tmp_path):
    from ml.preprocessing import gfs_extract

    class EmptyDataset:
        data_vars = {}
        coords = {"latitude": np.array([0.0]), "longitude": np.array([0.0])}
        variables = coords
        dims = {"latitude": 1, "longitude": 1}

        def close(self):
            pass

    monkeypatch.setattr(gfs_extract, "open_gfs_subset", lambda path: EmptyDataset())
    path = tmp_path / "temperature.grib2"
    path.touch()
    result = gfs_extract.extract_gfs_point({"temperature": path}, 0, 0)

    assert result["temperature_C"] is None
    assert result["field_status"]["temperature"] == "unavailable"
    assert result["precipitation_mm"] is None


def test_gfs_longitude_uses_zero_to_360_convention():
    from ml.preprocessing.gfs_extract import _gfs_longitude

    class FakeCoordinate:
        values = np.array([0.0, 90.0, 180.0, 270.0])

    class FakeDataset:
        coords = {"longitude": FakeCoordinate()}

        def __getitem__(self, name):
            return self.coords[name]

    assert _gfs_longitude(FakeDataset(), -72.8777) == pytest.approx(287.1223)


def test_gfs_coordinate_detection_supports_latitude_longitude_and_lat_lon():
    from ml.preprocessing.gfs_extract import _coordinate_name

    class CoordinateDataset:
        coords = {"latitude": object(), "longitude": object()}
        variables = {}
        dims = {}

    class ShortCoordinateDataset:
        coords = {"lat": object(), "lon": object()}
        variables = {}
        dims = {}

    assert _coordinate_name(CoordinateDataset(), "latitude", "lat") == "latitude"
    assert _coordinate_name(CoordinateDataset(), "longitude", "lon") == "longitude"
    assert _coordinate_name(ShortCoordinateDataset(), "latitude", "lat") == "lat"
    assert _coordinate_name(ShortCoordinateDataset(), "longitude", "lon") == "lon"


def test_real_gfs_subset_extraction_when_paths_are_configured():
    subset_keys = ("temperature", "wind_u", "wind_v", "precipitation")
    environment_paths = {key: os.environ.get(f"WCAST_GFS_{key.upper()}_PATH") for key in subset_keys}
    if not all(environment_paths.values()):
        pytest.skip("verified GFS subset paths are not configured")
    paths = {key: Path(value) for key, value in environment_paths.items()}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("configured GFS subset fixture is not present")
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    from ml.preprocessing.gfs_extract import extract_gfs_point

    result = extract_gfs_point(paths, 19.0760, 72.8777, lead_hours=24)
    assert result["temperature_C"] is not None
    assert result["wind_speed_ms"] is not None
    assert result["precipitation_mm"] is not None
    assert result["precipitation_mm"] is None