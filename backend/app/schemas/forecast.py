from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class ForecastValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float
    rainfall: float
    wind_speed: float


class ModelWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gfs: float
    gefs: float
    baseline: float


class Extremes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heavy_rain: bool
    heat_wave: bool
    high_wind: bool
    risk_level: str


class ExtremesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heavy_rain: bool
    heat_wave: bool
    high_wind: bool
    risk_level: str


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: Location
    lead_hours: int = Field(..., gt=0)
    forecast: ForecastValues
    weights: ModelWeights
    regime: str
    extremes: Extremes


class WeightLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class WeightSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: WeightLocation
    lead_hours: int = Field(..., gt=0)
    weights: ModelWeights
    regime: str
