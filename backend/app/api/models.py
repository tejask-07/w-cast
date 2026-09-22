from fastapi import APIRouter

from app.schemas.model import ModelListResponse

router = APIRouter(prefix="/api")


@router.get("/models", response_model=ModelListResponse)
def get_models() -> ModelListResponse:
    return ModelListResponse(
        models=[
            {"id": "gfs", "name": "GFS", "type": "NWP"},
            {"id": "gefs", "name": "GEFS", "type": "Ensemble NWP"},
            {"id": "baseline", "name": "Statistical Baseline", "type": "Baseline"},
        ]
    )
