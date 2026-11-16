"""Data loading, splitting, windowing and inference.

Every function here reproduces a step of the original notebook so that the
artifact script, the verification script and the Streamlit app share one
implementation. Nothing in this module imports Streamlit or TensorFlow at
import time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from oil_dashboard import config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF line endings normalized to LF.

    Git may check a file out with CRLF on Windows and LF on Linux, so raw bytes
    differ between machines while the content is identical. Any other change,
    including a single value, still changes the hash.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_prices(path: Path) -> pd.DataFrame:
    """Notebook cell 4: parse dates and sort chronologically."""
    df = pd.read_csv(path)
    missing = {"Date", config.TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@dataclass(frozen=True)
class SplitBounds:
    """Row positions: train [0, train_end), val [train_end, val_end), test [val_end, n)."""

    n: int
    train_end: int
    val_end: int

    def label(self, position: int) -> str:
        if position < self.train_end:
            return "Train"
        if position < self.val_end:
            return "Validation"
        return "Test"


def split_bounds(n: int) -> SplitBounds:
    """Notebook cell 13: integer 70/15/15 split by position."""
    train_size = int(n * config.TRAIN_FRAC)
    val_size = int(n * config.VAL_FRAC)
    return SplitBounds(n=n, train_end=train_size, val_end=train_size + val_size)


def create_sequences(data: np.ndarray, time_step: int = config.TIME_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Notebook cell 13, verbatim logic."""
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def notebook_arrays(df: pd.DataFrame, scaler: Any | None = None) -> dict[str, Any]:
    """Reproduce notebook cell 13.

    With ``scaler=None`` a new MinMaxScaler is fit on the training rows only
    (used once, by the artifact script). Pass the saved scaler to rebuild the
    exact same arrays without refitting.
    """
    from sklearn.preprocessing import MinMaxScaler

    time_step = config.TIME_STEP
    values = df[config.FEATURES].copy()
    b = split_bounds(len(values))

    train_values = values.iloc[: b.train_end].values
    val_values = values.iloc[b.train_end - time_step : b.val_end].values
    test_values = values.iloc[b.val_end - time_step :].values

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_values)
    else:
        train_scaled = scaler.transform(train_values)
    val_scaled = scaler.transform(val_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = create_sequences(train_scaled, time_step)
    X_val, y_val = create_sequences(val_scaled, time_step)
    X_test, y_test = create_sequences(test_scaled, time_step)

    return {
        "scaler": scaler,
        "bounds": b,
        "X_train": X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
        "y_train": y_train,
        "X_val": X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
        "y_val": y_val,
        "X_test": X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
        "y_test": y_test,
        "val_dates": df["Date"].iloc[b.train_end : b.val_end].reset_index(drop=True),
        "test_dates": df["Date"].iloc[b.val_end :].reset_index(drop=True),
    }


def latest_window(df: pd.DataFrame, time_step: int = config.TIME_STEP) -> pd.DataFrame:
    """The last ``time_step`` observations: the input for the next-step forecast."""
    if len(df) < time_step:
        raise ValueError(f"Need at least {time_step} observations, found {len(df)}.")
    window = df[["Date", config.TARGET]].iloc[-time_step:].reset_index(drop=True)
    if window[config.TARGET].isna().any():
        raise ValueError("The latest input window contains missing Close values.")
    return window


def scale_window(close_values: np.ndarray, scaler: Any) -> np.ndarray:
    """Scale with the saved training scaler (transform only) and shape to [1, steps, 1]."""
    column = np.asarray(close_values, dtype="float64").reshape(-1, 1)
    return scaler.transform(column).reshape(1, column.shape[0], 1)


def predict_next(model: Any, scaler: Any, close_values: np.ndarray) -> float:
    """One-step-ahead forecast in USD, mirroring notebook cells 19/25."""
    x = scale_window(close_values, scaler)
    scaled_pred = model.predict(x, verbose=0)
    return float(scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0, 0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Same metric definitions as the notebook (sklearn MAE, sqrt MSE, R^2)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }
