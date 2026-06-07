"""Pydantic schemas for the prediction API.

Fields are placeholders — replace with your actual model features once defined.
Each field maps to a column the trained model expects as input.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    # ── Replace these with your real feature fields ───────────────────────────
    venue_capacity: int = Field(..., gt=0, example=5000)
    feature_b: float = Field(..., example=3.14)
    feature_c: str = Field(..., example="category_x")
    # ─────────────────────────────────────────────────────────────────────────

    model_config = {
        "json_schema_extra": {
            "example": {
                "venue_capacity": 5000,
                "feature_b": 3.14,
                "feature_c": "category_x",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_attendance: float