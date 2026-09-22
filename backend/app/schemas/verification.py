from pydantic import BaseModel, ConfigDict


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mae: float
    rmse: float
    bias: float


class VerificationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gfs: Metric
    gefs: Metric
    adaptive_blend: Metric


class VerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    lead_hours: int
    metrics: VerificationMetrics
