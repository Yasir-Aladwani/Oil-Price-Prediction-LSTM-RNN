"""Cached loading of the dataset and the saved artifacts for the Streamlit app.

The app only reads files here; it never fits a scaler or trains a model.
Cache keys include each file's modification time, so regenerated artifacts
are picked up without restarting the server.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from oil_dashboard import config, pipeline


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


# ---------------------------------------------------------------- dataset ---
@st.cache_data(show_spinner="Loading price history...")
def _load_prices(path: str, mtime: float) -> pd.DataFrame:
    return pipeline.load_prices(Path(path))


@st.cache_data(show_spinner=False)
def _sha256(path: str, mtime: float) -> str:
    return pipeline.file_sha256(Path(path))


def prices() -> pd.DataFrame:
    """Return the dataset or stop the page with instructions."""
    path = config.DATA_PATH
    if not path.exists():
        st.error(
            f"Place `Crude_Oil_Data.csv` in the `data/` folder (expected at `{path}`). "
            f"Download it from Kaggle: {config.DATASET_URL}",
            title="Dataset not found",
        )
        st.stop()
    try:
        return _load_prices(str(path), _mtime(path))
    except Exception as exc:  # malformed file
        st.error(f"`{path.name}` could not be read: {exc}", title="Dataset could not be loaded")
        st.stop()


def data_sha256() -> str | None:
    path = config.DATA_PATH
    return _sha256(str(path), _mtime(path)) if path.exists() else None


# -------------------------------------------------------------- artifacts ---
@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=list(parse_dates))


@st.cache_resource(show_spinner="Loading model...", max_entries=4)
def _load_model(path: str, mtime: float) -> Any:
    import keras

    return keras.saving.load_model(path, compile=False)


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_scaler(path: str, mtime: float) -> Any:
    import joblib

    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def tensorflow_installed() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


@dataclass
class Artifacts:
    directory: Path
    metadata: dict[str, Any] | None = None
    predictions: pd.DataFrame | None = None
    history: pd.DataFrame | None = None
    comparison: pd.DataFrame | None = None  # metrics table saved by the notebook
    model_paths: dict[str, Path] = field(default_factory=dict)
    scaler_path: Path | None = None
    problems: list[str] = field(default_factory=list)

    def missing_for_inference(self, model_name: str) -> list[str]:
        """Requirements that block a live forecast with ``model_name``."""
        missing = []
        if model_name not in self.model_paths:
            missing.append(f"`{config.MODEL_FILES[model_name]}` (trained {model_name} model)")
        elif self.verified_sha(model_name) is False:
            missing.append(f"`{config.MODEL_FILES[model_name]}` matching the verified Colab model (this file differs)")
        if self.scaler_path is None:
            missing.append(f"`{config.SCALER_FILE}` (training scaler)")
        elif self.verified_sha("scaler") is False:
            missing.append(f"`{config.SCALER_FILE}` matching the verified Colab scaler (this file differs)")
        if not tensorflow_installed():
            missing.append("the `tensorflow` package")
        return missing

    def verified_sha(self, item: str) -> bool | None:
        """Whether a model (by name) or "scaler" still matches the file verified at import."""
        if not self.metadata:
            return None
        if item == "scaler":
            path, expected = self.scaler_path, self.metadata.get("source", {}).get("scaler_sha256")
        else:
            path = self.model_paths.get(item)
            expected = self.metadata.get("training", {}).get("models", {}).get(item, {}).get("sha256")
        if path is None or expected is None:
            return None
        return _sha256(str(path), _mtime(path)) == expected

    def data_matches(self, current_sha: str | None) -> bool | None:
        if not self.metadata or current_sha is None:
            return None
        return self.metadata.get("data", {}).get("sha256") == current_sha


def artifacts() -> Artifacts:
    d = config.ARTIFACTS_DIR
    art = Artifacts(directory=d)
    meta_path = d / config.METADATA_FILE
    if meta_path.exists():
        try:
            art.metadata = _read_json(str(meta_path), _mtime(meta_path))
        except Exception as exc:
            art.problems.append(f"`{config.METADATA_FILE}` is unreadable: {exc}")
    pred_path = d / config.PREDICTIONS_FILE
    if pred_path.exists():
        try:
            art.predictions = _read_csv(str(pred_path), _mtime(pred_path), ("Date",))
        except Exception as exc:
            art.problems.append(f"`{config.PREDICTIONS_FILE}` is unreadable: {exc}")
    comp_path = d / config.COMPARISON_FILE
    if comp_path.exists():
        try:
            art.comparison = _read_csv(str(comp_path), _mtime(comp_path)).set_index("Model")
        except Exception as exc:
            art.problems.append(f"`{config.COMPARISON_FILE}` is unreadable: {exc}")
    hist_path = d / config.HISTORY_FILE
    if hist_path.exists():
        art.history = _read_csv(str(hist_path), _mtime(hist_path))
    for name, file_name in config.MODEL_FILES.items():
        if (d / file_name).exists():
            art.model_paths[name] = d / file_name
    if (d / config.SCALER_FILE).exists():
        art.scaler_path = d / config.SCALER_FILE
    return art


def load_model(path: Path) -> Any:
    return _load_model(str(path), _mtime(path))


def load_scaler(path: Path) -> Any:
    return _load_scaler(str(path), _mtime(path))


def artifact_fingerprint(art: Artifacts, model_name: str) -> str:
    """Identifies the exact model + scaler files a forecast was produced with."""

    def stamp(path: Path | None) -> str:
        if path is None or not path.exists():
            return "none"
        return f"{path.name}:{path.stat().st_size}:{_mtime(path)}"

    return f"{model_name}={stamp(art.model_paths.get(model_name))}|scaler={stamp(art.scaler_path)}"


SETUP_COMMAND = "python scripts/import_colab_artifacts.py"
