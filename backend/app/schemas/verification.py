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
    equal_blend: Metric | None = None
    wcast: Metric | None = None
    improvement_vs_gfs: float | None = None
    improvement_vs_gefs: float | None = None


class VerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    lead_hours: int
    metrics: VerificationMetrics
