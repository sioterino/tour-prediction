"""Central configuration: paths, dataset registry, and shared constants."""

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path("E:/code/tour-prediction")

DATA_DIR        = ROOT / "data"
PROCESSED_DIR   = DATA_DIR / "processed"
RESULTS_DIR     = DATA_DIR / "results"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR      = ROOT / "models"

METRICS_PATH = RESULTS_DIR / "metrics.csv"

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS: dict[str, Path] = {
    "v1_baseline":       PROCESSED_DIR / "attendance_v1_baseline.csv",
    "v2_artist":         PROCESSED_DIR / "attendance_v2_artist.csv",
    "v3_artist_geo":     PROCESSED_DIR / "attendance_v3_artist_geo.csv",
    "v4_artist_geo_time":PROCESSED_DIR / "attendance_v4_artist_geo_time.csv",
    "v5_full":           PROCESSED_DIR / "attendance_v5_full.csv",
    "v6_no_artist":      PROCESSED_DIR / "attendance_v6_no_artist.csv",
}

# ── Feature engineering ───────────────────────────────────────────────────────
# Columns that leak target information and must be dropped before modelling.
LEAKY_COLS: list[str] = [
    "fill_rate",
    "box_score",
    "avg_ticket_price",
]

# ── Exporting ─────────────────────────────────────────────────────────────────
MODEL_EXT = '.joblib'
