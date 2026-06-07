from pandas import DataFrame

from src.preprocess.experiments.common import (
    add_artist_age_bucket,
    add_artist_country_interaction,
    add_show_time_features,
    add_weekend_flag,
    add_years_since_debut,
    drop_leakage_columns,
    parse_dates,
    safe_select_columns,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_v4_artist_geo_time(dataframe: DataFrame) -> DataFrame:
    logger.info("Starting V4 artist+geo+time preprocessing")
    dataframe = parse_dates(dataframe)
    dataframe = add_years_since_debut(dataframe)
    dataframe = add_artist_age_bucket(dataframe)
    dataframe = add_artist_country_interaction(dataframe)
    dataframe = add_show_time_features(dataframe)
    dataframe = add_weekend_flag(dataframe)
    dataframe = drop_leakage_columns(dataframe)

    columns = [
        "attendance",
        "reporting_status",
        "venue_name",
        "venue_type",
        "venue_capacity",
        "continent",
        "country",
        "city",
        "show_nights",
        "artist_name",
        "gender",
        "generation",
        "members",
        "company",
        "years_since_debut",
        "artist_age_bucket",
        "artist_country",
        "show_year",
        "show_month",
        "show_quarter",
        "weekday_1",
        "weekday_n",
        "is_weekend",
    ]

    processed = safe_select_columns(dataframe, columns)
    logger.info("V4 artist+geo+time preprocessing completed")
    return processed
