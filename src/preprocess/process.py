from pandas import DataFrame

from src.preprocess.experiments import (
    v1_baseline,
    v2_artist,
    v3_artist_geo,
    v4_artist_geo_time,
    v5_full,
    v6_no_artist,
)
from src.utils.data import load_tour_dataset
from src.utils.logger import setup_logger
from src.utils.save import save_processed_dataset

logger = setup_logger(__name__)


def main() -> None:
    logger.info("Loading raw tour dataset")
    dataset = load_tour_dataset()

    experiments: list[tuple[str, callable[[DataFrame], DataFrame]]] = [
        ("attendance_v1_baseline.csv", v1_baseline.process_v1_baseline),
        ("attendance_v2_artist.csv", v2_artist.process_v2_artist),
        ("attendance_v3_artist_geo.csv", v3_artist_geo.process_v3_artist_geo),
        ("attendance_v4_artist_geo_time.csv", v4_artist_geo_time.process_v4_artist_geo_time),
        ("attendance_v5_full.csv", v5_full.process_v5_full),
        ("attendance_v6_no_artist.csv", v6_no_artist.process_v6_no_artist),
    ]

    for output_name, processor in experiments:
        logger.info("Running experiment: %s", output_name)
        processed = processor(dataset)
        save_processed_dataset(processed, output_name)

    logger.info("All preprocessing experiments finished")


if __name__ == "__main__":
    main()
