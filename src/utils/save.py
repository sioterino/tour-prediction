from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.utils.logger import setup_logger


logger = setup_logger()


def save_dataset(dataframe: DataFrame, output_path: str | Path, file_name: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        dataframe: DataFrame to save.
        output_path: Directory where the file will be saved.
        file_name: Name of the output CSV file.
    """
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    if not file_name.endswith(".csv"):
        file_name += ".csv"

    file_path = path / file_name

    logger.info("Saving dataset to: %s", file_path)

    try:
        dataframe.to_csv(file_path, index=False)
    except Exception:
        logger.exception("Failed to save dataset.")
        exit(1)

    logger.info(
        "Dataset saved successfully: %d rows, %d columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )


def save_processed_dataset(dataframe: DataFrame, file_name: str) -> None:
    """
    Save a processed dataset to the predefined processed-data directory.

    Args:
        dataframe: DataFrame to save.
        file_name: Name of the output CSV file.
    """
    processed_path = Path("E:\\code\\tour-prediction\\data\\processed")

    try:
        save_dataset(
            dataframe=dataframe,
            output_path=processed_path,
            file_name=file_name,
        )
    except Exception as error:
        logger.error("Failed to save dataset: %s", error)
        exit(1)