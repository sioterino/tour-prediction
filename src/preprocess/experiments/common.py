from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


WEEKEND_VALUES = {"Fri", "Sat", "Sun"}


def parse_dates(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    dataframe["day_1"] = pd.to_datetime(dataframe["day_1"], errors="coerce")
    dataframe["debut_date"] = pd.to_datetime(dataframe["debut_date"], errors="coerce")
    return dataframe


def add_years_since_debut(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    if "day_1" not in dataframe or "debut_date" not in dataframe:
        return dataframe

    delta_years = dataframe["day_1"].dt.year - dataframe["debut_date"].dt.year
    birthday_passed = (
        (dataframe["day_1"].dt.month > dataframe["debut_date"].dt.month)
        | (
            (dataframe["day_1"].dt.month == dataframe["debut_date"].dt.month)
            & (dataframe["day_1"].dt.day >= dataframe["debut_date"].dt.day)
        )
    )
    dataframe["years_since_debut"] = delta_years.where(birthday_passed, delta_years - 1)
    return dataframe


def add_artist_age_bucket(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()

    def bucket(value: float) -> str | pd.NA: # type: ignore
        if pd.isna(value):
            return pd.NA
        try:
            years = float(value)
        except (TypeError, ValueError):
            return pd.NA

        if years <= 2:
            return "rookie"
        if years <= 5:
            return "growing"
        if years <= 10:
            return "established"
        return "legacy"

    dataframe["artist_age_bucket"] = dataframe["years_since_debut"].apply(bucket)
    return dataframe


def add_weekend_flag(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    dataframe["is_weekend"] = dataframe["weekday_1"].isin(WEEKEND_VALUES).astype(int)
    return dataframe


def add_show_time_features(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    dataframe["show_year"] = dataframe["day_1"].dt.year
    dataframe["show_month"] = dataframe["day_1"].dt.month
    dataframe["show_quarter"] = dataframe["day_1"].dt.quarter
    return dataframe


def get_country_hemisphere(country: str) -> str:
    if not isinstance(country, str):
        return "north"

    country = country.strip()

    logger.warning("Unknown hemisphere for country '%s', defaulting to northern hemisphere", country)
    return "north"


def add_artist_country_interaction(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    dataframe["artist_country"] = (
        dataframe["artist_name"].astype(str).fillna("")
        + "_"
        + dataframe["country"].astype(str).fillna("")
    )
    return dataframe


def drop_leakage_columns(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.copy()
    columns_to_drop = ["fill_rate", "box_score", "avg_ticket_price"]
    return dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns])


def safe_select_columns(dataframe: DataFrame, columns: list[str]) -> DataFrame:
    return dataframe.reindex(columns=columns)
