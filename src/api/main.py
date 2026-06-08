"""FastAPI application entrypoint.

Run with:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router

app = FastAPI(
    title="Tour Attendance Prediction API",
    description="Returns a predicted attendance value given event features.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return { "status": "ok" }
