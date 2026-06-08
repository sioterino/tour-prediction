"""Extract unique venue records from the raw dataset."""

from pandas import DataFrame


def extract_venues(df: DataFrame) -> list[dict]:
    """Return one record per venue with location and capacity info.

    Rows with a null ``venue_name`` or ``venue_capacity`` are dropped —
    the model requires capacity and unnamed venues can't be selected in the UI.

    The result is sorted continent → country → city → venue_name so the
    frontend can build cascading selects without any client-side sorting.
    """
    venues = (
        df[["venue_name", "venue_type", "venue_capacity", "continent", "country", "city"]]
        .drop_duplicates()
        .dropna(subset=["venue_name", "venue_capacity"])
        .sort_values(["continent", "country", "city", "venue_name"])
        .reset_index(drop=True)
    )
    venues.insert(0, "id", venues.index + 1)
    venues["venue_capacity"] = venues["venue_capacity"].astype(int)
    return venues.to_dict(orient="records")