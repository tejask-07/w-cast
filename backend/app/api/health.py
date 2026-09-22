from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> dict[str, str]:
    return {"status": "ok"}
