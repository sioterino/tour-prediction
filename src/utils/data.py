from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.utils.logger import setup_logger


logger = setup_logger()


def load_dataset(csv_path: str | Path) -> DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If csv_path is empty.
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """
    path = Path(csv_path)

    if not path.exists():
        logger.error(f"Dataset file not found: {path}")
        exit(1)

    if not path.is_file():
        logger.error(f"Dataset path is not a file: {path}")
        exit(1)

    logger.info("Loading dataset from: %s", path)

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.exception("The dataset file is empty.")
        exit(1)
    except pd.errors.ParserError:
        logger.exception("Failed to parse the CSV file.")
        exit(1)

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


def load_tour_dataset() -> DataFrame:
    dataset_path = "E:\\code\\tour-prediction\\data\\raw\\dataset.csv"

    try:
        return load_dataset(dataset_path)
    except Exception as error:
        logger.error("Failed to load dataset: %s", error)
        exit(1)


if __name__ == "__main__":
    dataset = load_tour_dataset()
    print(dataset.head())
    