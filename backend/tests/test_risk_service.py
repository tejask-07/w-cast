import numpy as np

from app.services import risk_service
from app.services.risk_service import calculate_weather_risk


def test_weather_risk_is_bounded_and_responds_to_forecast_inputs():
    temperature = np.array([[25.0, 40.0]])
    rainfall = np.array([[0.0, 100.0]])
    wind_speed = np.array([[0.0, 30.0]])
    no_disagreement = np.zeros((1, 2))

    scores, confidence = calculate_weather_risk(
        temperature,
        rainfall,
        wind_speed,
        no_disagreement,
        no_disagreement,
        no_disagreement,
        24,
    )

    assert 0 <= scores[0, 0] < scores[0, 1] <= 100
    assert np.all(confidence == 100)


def test_weather_risk_confidence_decreases_with_model_disagreement():
    values = np.array([[30.0]])
    no_disagreement = np.zeros((1, 1))
    disagreement = np.array([[15.0]])

    _, clear_confidence = calculate_weather_risk(
        values,
        values,
        values,
        no_disagreement,
        no_disagreement,
        no_disagreement,
        24,
    )
    risk, uncertain_confidence = calculate_weather_risk(
        values,
        values,
        values,
        disagreement,
        disagreement,
        disagreement,
        24,
    )

    assert uncertain_confidence[0, 0] < clear_confidence[0, 0]
    assert 0 <= risk[0, 0] <= 100


def test_spatial_risk_grid_uses_aoi_bounds_and_varies_by_cell(monkeypatch):
    monkeypatch.setattr(
        risk_service,
        "_download_forecast_sources",
        lambda lead_hours: ({"model": "gfs"}, {"model": "gefs"}),
    )
    monkeypatch.setattr(
        risk_service,
        "_weights",
        lambda *args: {"gfs": 0.5, "gefs": 0.5},
    )

    def sample_field(paths, variable, latitude, longitude, bounds, grid_latitudes, grid_longitudes):
        grid_lat, grid_lon = np.meshgrid(grid_latitudes, grid_longitudes, indexing="ij")
        offset = 0.0 if paths["model"] == "gfs" else 1.0
        if variable == "temperature":
            return 27.0 + (grid_lat - 19.0) * 10 + offset
        if variable == "rainfall":
            return (grid_lon - 72.5) * 80 + offset
        return 5.0 + (grid_lat - 19.0) * 3 + offset

    monkeypatch.setattr(risk_service, "_sample_model_field", sample_field)

    result = risk_service.generate_spatial_risk_grid(
        latitude=19.25,
        longitude=72.75,
        lead_hours=24,
        south=19.0,
        north=19.5,
        west=72.5,
        east=73.0,
    )

    assert result["bounds"] == [[19.0, 72.5], [19.5, 73.0]]
    assert len(result["cells"]) == 144
    assert len({cell["south"] for cell in result["cells"]}) == 12
    assert len({cell["west"] for cell in result["cells"]}) == 12
    assert len({cell["risk_score"] for cell in result["cells"]}) > 1
    assert all(19.0 <= cell["south"] < cell["north"] <= 19.5 for cell in result["cells"])
    assert all(72.5 <= cell["west"] < cell["east"] <= 73.0 for cell in result["cells"])