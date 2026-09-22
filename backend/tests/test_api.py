from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_ok():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_forecast_valid_request():
    response = client.get('/api/forecast', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 24, 'variable': 'rainfall'})
    assert response.status_code == 200
    data = response.json()
    assert data['location']['name'] == 'Mumbai'
    assert data['lead_hours'] == 24
    assert 'forecast' in data
    assert 'weights' in data
    assert 'regime' in data
    assert 'extremes' in data


def test_invalid_latitude_rejected():
    response = client.get('/api/forecast', params={'lat': -91, 'lon': 72.87, 'lead_hours': 24, 'variable': 'rainfall'})
    assert response.status_code in {400, 422}


def test_invalid_longitude_rejected():
    response = client.get('/api/forecast', params={'lat': 19.07, 'lon': 181, 'lead_hours': 24, 'variable': 'rainfall'})
    assert response.status_code in {400, 422}


def test_invalid_lead_hours_rejected():
    response = client.get('/api/forecast', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 0, 'variable': 'rainfall'})
    assert response.status_code in {400, 422}


def test_unsupported_variable_rejected():
    response = client.get('/api/forecast', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 24, 'variable': 'humidity'})
    assert response.status_code in {400, 422}


def test_models_endpoint_returns_expected_list():
    response = client.get('/api/models')
    assert response.status_code == 200
    data = response.json()
    assert 'models' in data
    ids = [item['id'] for item in data['models']]
    assert 'gfs' in ids
    assert 'gefs' in ids
    assert 'baseline' in ids


def test_weights_endpoint_returns_expected_structure():
    response = client.get('/api/weights', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 24})
    assert response.status_code == 200
    data = response.json()
    assert data['location']['lat'] == 19.07
    assert data['location']['lon'] == 72.87
    assert data['lead_hours'] == 24
    assert set(data['weights']) == {'gfs', 'gefs', 'baseline'}
    assert data['regime'] in {'wet', 'dry', 'moderate'}


def test_extremes_endpoint_returns_expected_structure():
    response = client.get('/api/extremes', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 24})
    assert response.status_code == 200
    data = response.json()
    assert 'heavy_rain' in data
    assert 'heat_wave' in data
    assert 'high_wind' in data
    assert 'risk_level' in data


def test_verification_endpoint_returns_expected_metrics():
    response = client.get('/api/verification', params={'lat': 19.07, 'lon': 72.87, 'lead_hours': 24, 'variable': 'rainfall'})
    assert response.status_code == 200
    data = response.json()
    assert data['variable'] == 'rainfall'
    assert data['lead_hours'] == 24
    assert 'gfs' in data['metrics']
    assert 'gefs' in data['metrics']
    assert 'adaptive_blend' in data['metrics']
    assert set(data['metrics']['gfs']) == {'mae', 'rmse', 'bias'}
