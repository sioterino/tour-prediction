from pandas import DataFrame

from src.utils.logger import setup_logger
from src.utils.data import load_tour_dataset
from src.utils.save import save_processed_dataset


logger = setup_logger()

def process_artist_only(dataframe: DataFrame) -> DataFrame:
    return dataframe


if __name__ == "__main__":
    dataset = load_tour_dataset()
    processed = process_artist_only(dataset)
    print(processed.head())
    save_processed_dataset(processed, 'test')
