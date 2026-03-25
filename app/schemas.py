"""Pydantic schemas for request/response validation."""

from enum import StrEnum

from pydantic import BaseModel, Field


class SexEnum(StrEnum):
    male = "male"
    female = "female"


class IslandEnum(StrEnum):
    torgersen = "Torgersen"
    biscoe = "Biscoe"
    dream = "Dream"


class PenguinFeatures(BaseModel):
    """Input features for penguin species prediction."""

    bill_length_mm: float = Field(..., ge=25.0, le=65.0, description="Bill length in millimeters")
    bill_depth_mm: float = Field(..., ge=12.0, le=22.0, description="Bill depth in millimeters")
    flipper_length_mm: float = Field(
        ..., ge=170.0, le=240.0, description="Flipper length in millimeters"
    )
    body_mass_g: float = Field(..., ge=2500.0, le=6500.0, description="Body mass in grams")
    sex: SexEnum = Field(..., description="Penguin sex")
    island: IslandEnum = Field(..., description="Island where the penguin was observed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bill_length_mm": 39.1,
                    "bill_depth_mm": 18.7,
                    "flipper_length_mm": 181.0,
                    "body_mass_g": 3750.0,
                    "sex": "male",
                    "island": "Torgersen",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: str
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_loaded: bool
    model_version: str | None
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    model_version: str
    model_type: str
    training_date: str
    accuracy: float
    features: list[str]
    target_classes: list[str]
    pipeline_sha256: str
