from pandas import DataFrame

from src.preprocess.experiments.common import (
    add_artist_country_interaction,
    add_artist_age_bucket,
    add_years_since_debut,
    drop_leakage_columns,
    parse_dates,
    safe_select_columns,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_v3_artist_geo(dataframe: DataFrame) -> DataFrame:
    logger.info("Starting V3 artist+geo preprocessing")
    dataframe = parse_dates(dataframe)
    dataframe = add_years_since_debut(dataframe)
    dataframe = add_artist_age_bucket(dataframe)
    dataframe = add_artist_country_interaction(dataframe)
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
    ]

    processed = safe_select_columns(dataframe, columns)
    logger.info("V3 artist+geo preprocessing completed")
    return processed
