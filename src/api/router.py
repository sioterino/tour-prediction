"""Prediction router."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api import service

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        value = service.predict(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(predicted_attendance=value)