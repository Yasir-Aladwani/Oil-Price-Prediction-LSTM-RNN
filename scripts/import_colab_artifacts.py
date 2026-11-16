"""Verify the Google Colab run's models and install them for the dashboard.

Nothing is trained or refit here. The script:

1. reads the Colab export in colab_export/ (two .keras models, the fitted
   scaler and model_comparison.csv) and the executed notebook, which is kept
   in the project root (use --notebook for another copy);
2. rebuilds the notebook's arrays from data/Crude_Oil_Data.csv with the
   exported scaler (transform only) and checks the shapes, split and scaler
   range against the notebook's printed outputs;
3. recomputes test-period predictions and MAE / RMSE / R^2 with the saved
   models and compares them with model_comparison.csv (full precision);
4. matches each model's validation loss to the per-epoch training log in the
   notebook and checks that the saved weights come from the best epoch;
5. adds the persistence baseline on the same test targets;
6. if every check passes, backs up the current artifacts/ folder to
   backups/ (ignored by Git) and installs the verified files in artifacts/.

    python scripts/import_colab_artifacts.py            # verify and install
    python scripts/import_colab_artifacts.py --check    # verify only
    python scripts/import_colab_artifacts.py --backup-name my_label
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oil_dashboard import config, pipeline  # noqa: E402

NOTEBOOK_NAME = "Oil_Price_Prediction_LSTM_RNN_Comparison.ipynb"
BACKUP_DIR = ROOT / "backups"

# Float32 inference on different hardware and TensorFlow builds changes
# predictions by about 1e-6 USD. These limits are far above that noise and far
# below any difference that would change a conclusion (0.001 USD is under
# 0.05% of either model's RMSE).
TOL = {"MAE": 1e-3, "RMSE": 1e-3, "R2": 1e-5}

results: list[tuple[str, bool, str]] = []
warnings_out: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def _text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def notebook_cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def cell_stdout(cell: dict) -> str:
    out = "".join(_text(o["text"]) for o in cell.get("outputs", []) if o.get("output_type") == "stream")
    return re.sub(r"\x1b\[[0-9;]*m", "", out)  # strip ANSI colors from Keras progress bars


def find_cell(cells: list[dict], marker: str) -> dict:
    for cell in cells:
        if cell["cell_type"] == "code" and marker in _text(cell["source"]):
            return cell
    raise LookupError(f"No notebook code cell contains {marker!r}")


def logged_history(cell: dict) -> pd.DataFrame:
    pairs = re.findall(r"loss: ([0-9.e+-]+) - val_loss: ([0-9.e+-]+)", cell_stdout(cell))
    return pd.DataFrame(
        [(i + 1, float(a), float(b)) for i, (a, b) in enumerate(pairs)],
        columns=["Epoch", "Train loss", "Validation loss"],
    )


def keras_file_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("metadata.json"))


def pickled_sklearn_version(path: Path) -> str | None:
    """scikit-learn version recorded inside the pickled estimator (its `_sklearn_version` field)."""
    data = path.read_bytes()
    match = re.search(rb"_sklearn_version.{0,2}?\x8c(.)", data, re.DOTALL)  # SHORT_BINUNICODE: length byte + text
    if not match:
        return None
    start = match.end()
    return data[start : start + match.group(1)[0]].decode("ascii", "replace")


def notebook_versions(cells: list[dict]) -> dict[str, str]:
    """Library versions and seed printed by the notebook's setup cell ("Name : value" lines)."""
    try:
        out = cell_stdout(find_cell(cells, "set_random_seed("))
    except LookupError:
        return {}
    return {k.strip(): v.strip() for k, v in re.findall(r"^([A-Za-z][\w -]*?)\s*:\s*(\S+)\s*$", out, re.M)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "colab_export", help="Folder with the Colab export")
    parser.add_argument(
        "--notebook", type=Path, default=ROOT / NOTEBOOK_NAME, help="Executed notebook from the same Colab run (default: project root)"
    )
    parser.add_argument("--check", action="store_true", help="Verify only; do not change artifacts/")
    parser.add_argument("--backup-name", default=None, help="Folder name under backups/ for the replaced artifacts")
    args = parser.parse_args()
    src: Path = args.source

    notebook: Path = args.notebook
    required = [*config.MODEL_FILES.values(), config.SCALER_FILE, config.COMPARISON_FILE]
    missing = [name for name in required if not (src / name).exists()]
    if not notebook.exists():
        missing.append(str(notebook))
    if missing:
        print(f"Missing in {src}: {', '.join(missing)}")
        return 1
    if not config.DATA_PATH.exists():
        print(f"Dataset not found: {config.DATA_PATH}")
        return 1

    import keras
    import sklearn
    import tensorflow as tf

    import joblib

    cells = notebook_cells(notebook)
    trained_with = notebook_versions(cells)
    df = pipeline.load_prices(config.DATA_PATH)
    scaler_sklearn_version = pickled_sklearn_version(src / config.SCALER_FILE)
    if scaler_sklearn_version and scaler_sklearn_version != sklearn.__version__:
        # Not suppressed: scikit-learn also emits InconsistentVersionWarning on load.
        warnings_out.append(
            f"scaler.joblib was saved with scikit-learn {scaler_sklearn_version}; this environment has "
            f"{sklearn.__version__}. Install the version pinned in requirements.txt."
        )
    scaler = joblib.load(src / config.SCALER_FILE)
    a = pipeline.notebook_arrays(df, scaler)  # transform only, never fit
    b = a["bounds"]

    check(
        "Notebook ran without errors",
        not any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])),
        f"{sum(c['cell_type'] == 'code' for c in cells)} code cells",
    )
    if trained_with.get("scikit-learn"):
        check(
            "Scaler file and notebook report the same scikit-learn",
            scaler_sklearn_version == trained_with["scikit-learn"],
            f"scaler {scaler_sklearn_version}, notebook {trained_with['scikit-learn']}",
        )

    # --- Data, split and preprocessing match the Colab run ----------------------
    shape_out = cell_stdout(find_cell(cells, "Basic inspection"))
    check("Dataset shape matches the notebook", f"Shape: ({len(df)}, {df.shape[1]})" in shape_out, f"{len(df)} rows")
    seq_out = cell_stdout(find_cell(cells, "create_sequences"))
    expected = {k: tuple(a[k].shape) for k in ("X_train", "X_val", "X_test")}
    check(
        "Sequence shapes match the notebook",
        all(f"{k} shape: {v}" in seq_out for k, v in expected.items()),
        ", ".join(f"{k}={v}" for k, v in expected.items()),
    )
    train_close = df[config.TARGET].iloc[: b.train_end]
    check(
        "Exported scaler was fit on training rows only",
        np.isclose(scaler.data_min_[0], train_close.min()) and np.isclose(scaler.data_max_[0], train_close.max()),
        f"range {scaler.data_min_[0]:.4f} to {scaler.data_max_[0]:.4f}",
    )
    check("Exported scaler is a 0-1 MinMaxScaler on one feature", tuple(scaler.feature_range) == (0, 1) and scaler.n_features_in_ == 1)
    check(
        "Chronological split: train < validation < test",
        df["Date"].iloc[b.train_end - 1] < df["Date"].iloc[b.train_end] < df["Date"].iloc[b.val_end],
        f"test {df['Date'].iloc[b.val_end].date()} to {df['Date'].iloc[-1].date()}",
    )

    # --- Recompute predictions and metrics ------------------------------------
    reported = pd.read_csv(src / config.COMPARISON_FILE).set_index("Model")
    # The notebook must come from the same run as the export: its printed metrics
    # (full precision) must equal model_comparison.csv.
    printed = {"LSTM": "lstm_mae =", "Simple RNN": "rnn_mae ="}
    prefix = {"LSTM": "LSTM", "Simple RNN": "RNN"}
    for name, marker in printed.items():
        out = cell_stdout(find_cell(cells, marker))
        found = {
            k: float(m.group(1))
            for k, label in (("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²"))
            if (m := re.search(rf"{prefix[name]} {label}\s*:\s*([0-9.eE+-]+)", out))
        }
        check(
            f"{name}: notebook output matches model_comparison.csv",
            len(found) == 3 and all(found[k] == float(reported.loc[name, k]) for k in found),
            "same run" if len(found) == 3 else "metrics not found in the notebook output",
        )
    y_test = scaler.inverse_transform(a["y_test"].reshape(-1, 1)).ravel()
    test_dates = a["test_dates"].dt.tz_localize(None).dt.normalize()
    predictions = pd.DataFrame({"Date": test_dates, "Actual": y_test})
    close = df[config.TARGET].to_numpy()
    predictions["Persistence baseline"] = close[b.val_end - 1 : -1]  # previous actual close for each test target

    histories, model_info, metrics, diffs = [], {}, {}, {}
    fit_markers = {"LSTM": "lstm_model.fit(", "Simple RNN": "rnn_model.fit("}
    for name in config.MODEL_NAMES:
        path = src / config.MODEL_FILES[name]
        model = keras.saving.load_model(path, compile=False)
        pred = scaler.inverse_transform(model.predict(a["X_test"], verbose=0)).ravel()
        predictions[name] = pred
        metrics[name] = pipeline.regression_metrics(y_test, pred)
        diffs[name] = {k: abs(metrics[name][k] - float(reported.loc[name, k])) for k in ("MAE", "RMSE", "R2")}
        check(
            f"{name}: recomputed metrics match model_comparison.csv",
            all(diffs[name][k] <= TOL[k] for k in TOL),
            ", ".join(f"{k} {metrics[name][k]:.6f} (diff {diffs[name][k]:.1e})" for k in ("MAE", "RMSE", "R2")),
        )

        fit_cell = find_cell(cells, fit_markers[name])
        history = logged_history(fit_cell)
        history.insert(0, "Model", name)
        histories.append(history)
        best_epoch = int(history.loc[history["Validation loss"].idxmin(), "Epoch"])
        printed_run = re.search(r"epochs run: (\d+)", cell_stdout(fit_cell))
        printed_best = re.search(r"best epoch by val_loss: (\d+)", cell_stdout(fit_cell))
        if printed_run and printed_best:
            check(
                f"{name}: training log matches the notebook's summary",
                int(printed_run.group(1)) == len(history) and int(printed_best.group(1)) == best_epoch,
                f"{len(history)} epochs, best epoch {best_epoch}",
            )
        val_pred = model.predict(a["X_val"], verbose=0).ravel()
        val_mse = float(np.mean((val_pred - a["y_val"]) ** 2))
        rel = (history["Validation loss"] - val_mse).abs() / history["Validation loss"]
        weights_epoch = int(history.loc[rel.idxmin(), "Epoch"])
        check(
            f"{name}: saved weights match a logged epoch",
            rel.min() < 1e-3,
            f"validation MSE {val_mse:.4e} = epoch {weights_epoch} of {len(history)}",
        )
        check(
            f"{name}: saved weights are from the best epoch",
            weights_epoch == best_epoch,
            f"weights epoch {weights_epoch}, lowest validation loss at epoch {best_epoch}",
        )
        model_info[name] = {
            "file": path.name,
            "epochs_run": len(history),
            "best_logged_epoch": best_epoch,
            "best_logged_val_loss": float(history["Validation loss"].min()),
            "weights_epoch": weights_epoch,
            "weights_val_loss": val_mse,
            "parameters": int(model.count_params()),
            "keras_version_saved": keras_file_metadata(path).get("keras_version"),
            "date_saved": keras_file_metadata(path).get("date_saved"),
            "sha256": pipeline.file_sha256(path),
        }
    metrics["Persistence baseline"] = pipeline.regression_metrics(y_test, predictions["Persistence baseline"])

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    for message in warnings_out:
        print(f"[WARN] {message}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    if failed:
        print("Nothing was installed. Investigate the failed checks first.")
        return 1
    if args.check:
        return 0

    # --- Write the verified set to a staging folder, then swap it in ----------
    seed = trained_with.get("Random seed")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "kind": "colab",
            "description": "Google Colab training run recorded in the project notebook",
            "notebook": notebook.name,
            "notebook_sha256": pipeline.file_sha256(notebook),
            "scaler_sha256": pipeline.file_sha256(src / config.SCALER_FILE),
            "scaler_sklearn_version_saved": scaler_sklearn_version,
        },
        "data": {
            "file": config.DATA_PATH.name,
            "sha256": pipeline.text_sha256(config.DATA_PATH),
            "sha256_method": "SHA-256 after normalizing CRLF line endings to LF",
            "rows": len(df),
            "first_date": df["Date"].iloc[0].strftime("%Y-%m-%d"),
            "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        },
        "preprocessing": {
            "features": config.FEATURES,
            "target": config.TARGET,
            "time_step": config.TIME_STEP,
            "horizon": "1 observation ahead",
            "scaler": "MinMaxScaler(feature_range=(0, 1)) fit on training rows only",
            "scaler_data_min": float(scaler.data_min_[0]),
            "scaler_data_max": float(scaler.data_max_[0]),
        },
        "splits": {
            "train": {"rows": b.train_end, "start": df["Date"].iloc[0].strftime("%Y-%m-%d"),
                      "end": df["Date"].iloc[b.train_end - 1].strftime("%Y-%m-%d"), "targets": len(a["y_train"])},
            "validation": {"rows": b.val_end - b.train_end, "start": df["Date"].iloc[b.train_end].strftime("%Y-%m-%d"),
                           "end": df["Date"].iloc[b.val_end - 1].strftime("%Y-%m-%d"), "targets": len(a["y_val"])},
            "test": {"rows": b.n - b.val_end, "start": df["Date"].iloc[b.val_end].strftime("%Y-%m-%d"),
                     "end": df["Date"].iloc[-1].strftime("%Y-%m-%d"), "targets": len(a["y_test"])},
        },
        "training": {
            "environment": "Google Colab",
            "library_versions": {k: v for k, v in trained_with.items() if k != "Random seed"},
            "seed": int(seed) if seed and seed.isdigit() else None,
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "batch_size": 32,
            "max_epochs": 50,
            "early_stopping": "monitor=val_loss, patience=8, restore_best_weights=True (a separate instance per model)",
            "models": model_info,
        },
        "test_metrics": metrics,
        "reported_metrics": {name: {k: float(reported.loc[name, k]) for k in ("MAE", "RMSE", "R2")} for name in config.MODEL_NAMES},
        "verification": {
            "tolerance": TOL,
            "abs_diff": diffs,
            "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in results],
            "warnings": warnings_out,
        },
        "environment": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    target = config.ARTIFACTS_DIR
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in [*config.MODEL_FILES.values(), config.SCALER_FILE, config.COMPARISON_FILE]:
        shutil.copy2(src / name, staging / name)
    out = predictions.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(staging / config.PREDICTIONS_FILE, index=False)
    pd.concat(histories, ignore_index=True).to_csv(staging / config.HISTORY_FILE, index=False)
    (staging / config.METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if target.exists():
        BACKUP_DIR.mkdir(exist_ok=True)
        backup = BACKUP_DIR / (args.backup_name or f"artifacts_{datetime.now():%Y%m%d-%H%M%S}")
        if backup.exists():
            print(f"{backup.relative_to(ROOT)} already exists; choose another --backup-name.")
            shutil.rmtree(staging)
            return 1
        shutil.move(str(target), str(backup))
        print(f"Previous artifacts moved to {backup.relative_to(ROOT)}")
    shutil.move(str(staging), str(target))
    print(f"Installed the verified Colab artifacts in {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
