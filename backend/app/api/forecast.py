from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.schemas.forecast import (
    ForecastResponse,
    HourlyForecastResponse,
    SpatialForecastResponse,
    WeightSummaryResponse,
)
from app.services.forecast_service import (
    generate_extremes,
    generate_forecast,
    generate_weights,
    validate_variable,
)
from app.services.spatial_forecast_service import (
    generate_spatial_forecast_map,
    get_cached_map_path,
    validate_spatial_request,
)
from app.services.hourly_forecast_service import (
    generate_hourly_forecast,
    validate_hourly_request,
)

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

    if lead_hours not in (24, 48, 72):
        raise HTTPException(
            status_code=422,
            detail="lead_hours must be one of: 24, 48, 72",
        )

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


@router.get("/extremes")
def get_extremes(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(..., gt=0),
):
    try:
        return generate_extremes(
            lat=lat,
            lon=lon,
            lead_hours=lead_hours,
        )
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(
            status_code=503,
            detail="Extreme detection service unavailable",
        ) from exc


@router.get("/forecast/map", response_model=SpatialForecastResponse)
def get_forecast_map(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(...),
    variable: str = Query(...),
    south: float | None = Query(None, ge=-90, le=90),
    north: float | None = Query(None, ge=-90, le=90),
    west: float | None = Query(None, ge=-180, le=180),
    east: float | None = Query(None, ge=-180, le=180),
):
    try:
        validate_spatial_request(
            variable,
            lead_hours,
            south,
            north,
            west,
            east,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return generate_spatial_forecast_map(
            latitude=lat,
            longitude=lon,
            lead_hours=lead_hours,
            variable=variable,
            south=south,
            north=north,
            west=west,
            east=east,
        )
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Spatial forecast unavailable") from exc


@router.get("/forecast/map/image/{cache_name}")
def get_forecast_map_image(cache_name: str):
    try:
        image_path = get_cached_map_path(cache_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Forecast map image not found") from exc
    return FileResponse(image_path, media_type="image/png")


@router.get("/forecast/hourly", response_model=HourlyForecastResponse)
def get_hourly_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(...),
    variable: str = Query(...),
):
    try:
        validate_hourly_request(variable, lead_hours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return generate_hourly_forecast(
            latitude=lat,
            longitude=lon,
            lead_hours=lead_hours,
            variable=variable,
        )
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Hourly forecast unavailable") from exc
