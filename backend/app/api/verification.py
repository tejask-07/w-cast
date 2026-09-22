from fastapi import APIRouter, HTTPException, Query

from app.schemas.forecast import ExtremesResponse
from app.schemas.verification import VerificationResponse
from app.services.verification_service import generate_verification, validate_variable

router = APIRouter(prefix="/api")


@router.get("/verification", response_model=VerificationResponse)
def get_verification(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    variable: str = Query(...),
    lead_hours: int = Query(..., gt=0),
):
    try:
        validate_variable(variable)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return generate_verification(lat=lat, lon=lon, lead_hours=lead_hours, variable=variable)
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Verification service unavailable") from exc


@router.get("/extremes", response_model=ExtremesResponse)
def get_extremes(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lead_hours: int = Query(..., gt=0),
):
    try:
        from app.services.forecast_service import generate_extremes

        return generate_extremes(lat=lat, lon=lon, lead_hours=lead_hours)
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise HTTPException(status_code=503, detail="Extremes service unavailable") from exc
