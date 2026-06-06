from pandas import DataFrame

from src.preprocess.experiments.common import (
    add_artist_age_bucket,
    drop_leakage_columns,
    parse_dates,
    safe_select_columns,
    add_years_since_debut,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_v2_artist(dataframe: DataFrame) -> DataFrame:
    logger.info("Starting V2 artist preprocessing")
    dataframe = parse_dates(dataframe)
    dataframe = add_years_since_debut(dataframe)
    dataframe = add_artist_age_bucket(dataframe)
    dataframe = drop_leakage_columns(dataframe)

    columns = [
        "attendance",
        "reporting_status",
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
    ]

    processed = safe_select_columns(dataframe, columns)
    logger.info("V2 artist preprocessing completed")
    return processed
