from pandas import DataFrame

from src.preprocess.experiments.common import drop_leakage_columns, parse_dates, safe_select_columns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_v1_baseline(dataframe: DataFrame) -> DataFrame:
    logger.info("Starting V1 baseline preprocessing")
    dataframe = parse_dates(dataframe)
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
    ]

    processed = safe_select_columns(dataframe, columns)
    logger.info("V1 baseline preprocessing completed")
    return processed
