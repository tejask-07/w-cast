from fastapi import APIRouter, HTTPException, Query

from app.schemas.forecast import ForecastResponse, WeightSummaryResponse
from app.services.forecast_service import generate_forecast, generate_weights, validate_variable

router = APIRouter(prefix="/api")


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(..., gt=0),
    variable: str = Query(...),
):
    try:
        validate_variable(variable)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return generate_forecast(lat=lat, lon=lon, lead_hours=lead_hours, variable=variable)
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Forecast service unavailable") from exc


@router.get("/weights", response_model=WeightSummaryResponse, response_model_exclude_none=True)
def get_weights(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(..., gt=0),
):
    try:
        return generate_weights(lat=lat, lon=lon, lead_hours=lead_hours)
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Weight service unavailable") from exc
