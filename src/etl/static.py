"""Generate static JSON files consumed by the frontend.

Produces:
  data/static/artists.json  — one record per artist with all static attributes
  data/static/venues.json   — one record per venue with location and capacity

Run with:
    python -m src.generate_static
"""

from src.etl.artists import extract_artists
from src.etl.venues import extract_venues
from src.utils.data import load_tour_dataset
from src.utils.logger import setup_logger
from src.utils.save import save_processed_json

logger = setup_logger(__name__)


def main() -> None:
    logger.info("Loading raw dataset")
    df = load_tour_dataset()

    logger.info("Extracting artists")
    artists = extract_artists(df)
    save_processed_json(artists, "artists.json")
    logger.info("artists.json → %d records", len(artists))

    logger.info("Extracting venues")
    venues = extract_venues(df)
    save_processed_json(venues, "venues.json")
    logger.info("venues.json → %d records", len(venues))


if __name__ == "__main__":
    main()