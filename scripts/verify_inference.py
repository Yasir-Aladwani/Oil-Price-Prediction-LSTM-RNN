"""Check that the dashboard's inference matches the notebook's code path.

The notebook's own preprocessing cells (data loading, sorting, split, scaler
fit and windowing) are read from the .ipynb and executed unchanged, except for
the CSV path. Its inference step (``scaler.inverse_transform(model.predict(X))``)
is then applied with the saved model files and compared with:

1. the app's ``pipeline.predict_next`` on the same samples,
2. the saved test predictions shown on the Historical evaluation page,
3. the next-observation forecast shown on the Interactive forecast page.

It also re-checks the leakage guards and the saved metrics. Nothing is trained.

    python scripts/verify_inference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oil_dashboard import config, pipeline  # noqa: E402

NOTEBOOK = ROOT / "Oil_Price_Prediction_LSTM_RNN_Comparison.ipynb"
TOL_USD = 1e-4
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def notebook_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def main() -> int:
    import keras

    cells = notebook_cells()
    # Code cells in order: [0] imports + read_csv, [1] datetime + sort, ..., [6] scaling & sequences.
    load_src = cells[0].replace("/content/Crude_Oil_Data.csv", config.DATA_PATH.as_posix())
    prep_src = next(c for c in cells if "create_sequences" in c and "MinMaxScaler(" in c)
    ns: dict = {}
    exec(load_src, ns)
    exec(cells[1].replace("print(", "(lambda *a, **k: None)("), ns)
    exec(prep_src.replace("print(", "(lambda *a, **k: None)("), ns)
    nb_scaler, X_test, y_test, test_dates = ns["scaler"], ns["X_test"], ns["y_test"], ns["test_dates"]

    art = config.ARTIFACTS_DIR
    saved_scaler = joblib.load(art / config.SCALER_FILE)
    saved_pred = pd.read_csv(art / config.PREDICTIONS_FILE, parse_dates=["Date"])
    meta = json.loads((art / config.METADATA_FILE).read_text(encoding="utf-8"))
    colab = pd.read_csv(art / config.COMPARISON_FILE).set_index("Model")

    # Scaler: the notebook refits on train rows; the saved one must be identical.
    check(
        "Saved scaler equals the notebook's train-only scaler",
        np.allclose(saved_scaler.data_min_, nb_scaler.data_min_) and np.allclose(saved_scaler.data_max_, nb_scaler.data_max_),
        f"min={saved_scaler.data_min_[0]:.4f} max={saved_scaler.data_max_[0]:.4f}",
    )
    df = ns["df"]
    train_end = ns["train_size"]
    check(
        "Scaler range comes from training rows only",
        np.isclose(saved_scaler.data_max_[0], df["Close"].iloc[:train_end].max())
        and np.isclose(saved_scaler.data_min_[0], df["Close"].iloc[:train_end].min()),
    )
    val_dates = ns["val_dates"]
    check(
        "Chronological order: train < validation < test",
        df["Date"].iloc[train_end - 1] < val_dates.iloc[0] and val_dates.iloc[-1] < test_dates.iloc[0],
        f"test starts {test_dates.iloc[0].date()}",
    )
    check("Data file matches the one used for training", pipeline.file_sha256(config.DATA_PATH) == meta["data"]["sha256"])

    y_test_usd = nb_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    check(
        "Saved test dates and actual values align with the notebook",
        len(saved_pred) == len(test_dates)
        and (saved_pred["Date"].to_numpy() == test_dates.dt.tz_localize(None).dt.normalize().to_numpy()).all()
        and np.allclose(saved_pred["Actual"], y_test_usd, atol=1e-6),
        f"{len(saved_pred)} rows",
    )

    close = df["Close"].to_numpy()
    rng = np.random.default_rng(0)
    sample_idx = sorted({0, len(X_test) - 1, *rng.choice(len(X_test), 8, replace=False).tolist()})

    for name in config.MODEL_NAMES:
        model = keras.saving.load_model(art / config.MODEL_FILES[name], compile=False)
        # Notebook cells 19 / 25.
        nb_pred = nb_scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
        diff_saved = np.abs(nb_pred - saved_pred[name].to_numpy()).max()
        check(f"{name}: notebook predictions == saved test predictions", diff_saved < TOL_USD, f"max |diff| = {diff_saved:.2e} USD")

        # App path on the same samples (window of 60 raw closes before each test target).
        diffs = []
        for i in sample_idx:
            target_pos = ns["train_size"] + ns["val_size"] + i
            app_value = pipeline.predict_next(model, saved_scaler, close[target_pos - config.TIME_STEP : target_pos])
            diffs.append(abs(app_value - nb_pred[i]))
        check(f"{name}: app predict_next == notebook on {len(sample_idx)} test samples", max(diffs) < TOL_USD, f"max |diff| = {max(diffs):.2e} USD")

        # Next-observation forecast: notebook-style manual steps vs app function.
        manual_x = nb_scaler.transform(df[["Close"]].values[-config.TIME_STEP :]).reshape(1, config.TIME_STEP, 1)
        manual = float(nb_scaler.inverse_transform(model.predict(manual_x, verbose=0))[0, 0])
        app_forecast = pipeline.predict_next(model, saved_scaler, close[-config.TIME_STEP :])
        check(
            f"{name}: next-observation forecast matches notebook-style inference",
            abs(manual - app_forecast) < TOL_USD,
            f"app ${app_forecast:.4f} vs notebook-style ${manual:.4f}",
        )

        recomputed = pipeline.regression_metrics(saved_pred["Actual"], saved_pred[name])
        check(
            f"{name}: stored metrics match recomputation",
            all(np.isclose(recomputed[k], meta["test_metrics"][name][k]) for k in ("MAE", "RMSE", "R2")),
            ", ".join(f"{k}={v:.4f}" for k, v in recomputed.items()),
        )
        nb_metrics = pipeline.regression_metrics(y_test_usd, nb_pred)
        gap = max(abs(nb_metrics[k] - float(colab.loc[name, k])) for k in ("MAE", "RMSE", "R2"))
        check(f"{name}: notebook-path metrics match model_comparison.csv", gap < 1e-3, f"max |diff| = {gap:.1e}")

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
