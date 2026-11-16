"""Paths, settings and visual constants.

Preprocessing settings mirror `Oil_Price_Prediction_LSTM_RNN_Comparison.ipynb`
(section 4, "Scaling and Sequence Creation"). Change them only together with
retraining, because the saved models and scaler depend on them.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Environment overrides exist so the app can be pointed at other copies
# (and so missing-file handling can be tested without touching real files).
DATA_PATH = Path(os.environ.get("OIL_DATA_PATH", ROOT / "data" / "Crude_Oil_Data.csv"))
ARTIFACTS_DIR = Path(os.environ.get("OIL_ARTIFACTS_DIR", ROOT / "artifacts"))

# --- Notebook preprocessing (do not change without retraining) -------------
TIME_STEP = 60  # previous observations used to predict the next one
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = remainder (about 15%)
FEATURES = ["Close"]  # univariate model input and target
TARGET = "Close"

# --- Artifact file names -------------------------------------------------
# The models and scaler come from the Google Colab run recorded in the
# notebook. scripts/import_colab_artifacts.py verifies them and copies them
# here, keeping the file names the notebook saved them under.
MODEL_NAMES = ("LSTM", "Simple RNN")
MODEL_FILES = {
    "LSTM": "oil_price_lstm_model.keras",
    "Simple RNN": "oil_price_simple_rnn_model.keras",
}
SCALER_FILE = "scaler.joblib"
COMPARISON_FILE = "model_comparison.csv"  # metrics table written by the notebook
METADATA_FILE = "metadata.json"
PREDICTIONS_FILE = "test_predictions.csv"
HISTORY_FILE = "training_history.csv"

# RMSE gaps smaller than this (relative %) are reported as a tie, not a win.
NEGLIGIBLE_GAP_PCT = 2.0

REPO_URL = "https://github.com/Yasir-Aladwani/Oil-Price-Prediction-LSTM-RNN"
DATASET_URL = "https://www.kaggle.com/datasets/crude-oil-stock-dataset-2000-2024"

# --- Visual identity -------------------------------------------------------
# Inspired by blue/green energy-sector palettes; not an official brand palette.
COLORS = {
    "navy": "#123047",  # text and the "Actual" reference series
    "blue": "#007DBA",  # primary; LSTM series
    "green": "#78BE20",  # limited UI accent
    "green_line": "#5E9E1A",  # Simple RNN series (>= 3:1 on white, CVD-checked)
    "background": "#F5F8FA",
    "surface": "#FFFFFF",
    "border": "#DCE4EA",
    "grid": "#E8EEF2",
    "muted": "#5B6B78",
    "val_band": "#E9F2F8",
    "test_band": "#EEF6E4",
}
SERIES_COLORS = {
    "Actual": COLORS["navy"],
    "LSTM": COLORS["blue"],
    "Simple RNN": COLORS["green_line"],
}
FONT_FAMILY = "'Source Sans 3', 'Source Sans Pro', sans-serif"
