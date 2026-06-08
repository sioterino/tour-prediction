"""Pydantic schemas for the prediction API."""

from typing import Literal
from pydantic import BaseModel, Field


ArtistAgeBucket = Literal["rookie", "growing", "established", "legacy"]
VenueType = Literal["arena", "concert hall", "stadium", "theater", "amphitheater"]
Continent = Literal["Asia", "North America", "South America", "Europe", "Oceania", "Middle East"]


class PredictionRequest(BaseModel):
    # ── Venue (populated from venues.json via frontend) ───────────────────────
    venue_name:     str       = Field(..., example="Jamsil Indoor Stadium")
    venue_type:     VenueType = Field(..., example="arena")
    venue_capacity: int       = Field(..., gt=0, example=11069)
    continent:      Continent = Field(..., example="Asia")
    country:        str       = Field(..., example="South Korea")
    city:           str       = Field(..., example="Seoul")

    # ── Artist (populated from artists.json via frontend) ─────────────────────
    artist_name:       str            = Field(..., example="aespa")
    gender:            Literal["male", "female"] = Field(..., example="female")
    generation:        int            = Field(..., ge=1, le=5, example=4)
    members:           int            = Field(..., ge=1, example=4)
    company:           str            = Field(..., example="SM,KAKAO")
    years_since_debut: int            = Field(..., ge=0, example=2)
    artist_age_bucket: ArtistAgeBucket = Field(..., example="rookie")

    # ── User input ────────────────────────────────────────────────────────────
    show_nights: int = Field(..., ge=1, le=6, example=2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "venue_name":        "Jamsil Indoor Stadium",
                "venue_type":        "arena",
                "venue_capacity":    11069,
                "continent":         "Asia",
                "country":           "South Korea",
                "city":              "Seoul",
                "artist_name":       "aespa",
                "gender":            "female",
                "generation":        4,
                "members":           4,
                "company":           "SM,KAKAO",
                "years_since_debut": 2,
                "artist_age_bucket": "rookie",
                "show_nights":       2,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_attendance: int
