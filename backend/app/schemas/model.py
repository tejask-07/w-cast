from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo]
