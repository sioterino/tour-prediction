"""Extract unique artist records from the raw dataset."""

from pandas import DataFrame


def extract_artists(df: DataFrame) -> list[dict]:
    """Return one record per artist with all their static attributes.

    Includes ``debut_date`` so the frontend can calculate ``years_since_debut``
    and ``artist_age_bucket`` from the user-supplied show date without a
    round-trip to the API.
    """
    artists = (
        df[["artist_name", "gender", "generation", "members", "debut_date", "company"]]
        .drop_duplicates(subset=["artist_name"])
        .sort_values("artist_name")
        .reset_index(drop=True)
    )
    artists.insert(0, "id", artists.index + 1)
    return artists.to_dict(orient="records")