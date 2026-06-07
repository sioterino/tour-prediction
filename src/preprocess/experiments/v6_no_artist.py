from pandas import DataFrame

from src.preprocess.experiments.common import (
    add_artist_age_bucket,
    add_show_time_features,
    add_weekend_flag,
    add_years_since_debut,
    drop_leakage_columns,
    parse_dates,
    safe_select_columns,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_v6_no_artist(dataframe: DataFrame) -> DataFrame:
    logger.info("Starting V6 no-artist preprocessing")
    dataframe = parse_dates(dataframe)
    dataframe = add_years_since_debut(dataframe)
    dataframe = add_artist_age_bucket(dataframe)
    dataframe = add_show_time_features(dataframe)
    dataframe = add_weekend_flag(dataframe)
    dataframe = drop_leakage_columns(dataframe)

    columns = [
        "attendance",
        "reporting_status",
        "gender",
        "generation",
        "tour_type",
        "members",
        "stage_setup",
        "continent",
        "country",
        "city",
        "venue_name",
        "venue_type",
        "venue_capacity",
        "show_nights",
        "show_year",
        "show_month",
        "show_quarter",
        "weekday_1",
        "weekday_n",
        "is_weekend",
        "years_since_debut",
        "artist_age_bucket",
    ]

    processed = safe_select_columns(dataframe, columns)
    logger.info("V6 no-artist preprocessing completed")
    return processed
