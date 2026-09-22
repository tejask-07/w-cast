from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.forecast import router as forecast_router
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.verification import router as verification_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(forecast_router)
app.include_router(models_router)
app.include_router(verification_router)


@app.on_event("startup")
def startup_event() -> None:
    logger.info("Starting W-CAST backend")


@app.on_event("shutdown")
def shutdown_event() -> None:
    logger.info("Shutting down W-CAST backend")
