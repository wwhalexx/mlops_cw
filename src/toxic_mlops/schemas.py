"""Prediction response contract shared by the API and its HTTP client."""

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    label: int = Field(ge=0, le=1)
    label_name: str
    score: float = Field(ge=0, le=1)
    threshold: float
    model_version: str
    preprocessing_version: str
