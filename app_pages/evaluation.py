import numpy as np
import streamlit as st

from oil_dashboard import charts, config, pipeline, store, ui

df = store.prices()
art = store.artifacts()

st.header("Historical evaluation", anchor=False)

if art.predictions is None or art.metadata is None:
    st.warning(
        f"The saved test predictions (`artifacts/{config.PREDICTIONS_FILE}`) and `metadata.json` are missing. "
        f"Run `{store.SETUP_COMMAND}` once from the project folder to create them.",
        title="Saved results not found",
    )
    st.stop()

pred = art.predictions.copy()
meta = art.metadata
splits = meta["splits"]

# --- Integrity check: saved rows must line up with the dataset's test period.
bounds = pipeline.split_bounds(len(df))
test_rows = df.iloc[bounds.val_end :].reset_index(drop=True)
aligned = (
    len(test_rows) == len(pred)
    and (test_rows["Date"].dt.tz_localize(None).dt.normalize().to_numpy() == pred["Date"].to_numpy()).all()
    and np.allclose(test_rows["Close"].to_numpy(), pred["Actual"].to_numpy(), atol=1e-6)
)
if not aligned:
    st.error(
        "The saved predictions do not match the dates or prices in the current dataset. "
        f"Regenerate them with `{store.SETUP_COMMAND}`.",
        title="Saved results are out of sync with the dataset",
    )
    st.stop()

st.markdown(
    f"Each prediction uses the actual {config.TIME_STEP} closes before that day, so every point is a "
    "one-step-ahead forecast. The models never saw the test period during training or early stopping."
)

for col, (label, key) in zip(ui.kpi_row(3), (("Train", "train"), ("Validation", "validation"), ("Test", "test"))):
    s = splits[key]
    with col:
        ui.kpi(
            key,
            f"{label} period",
            f"{s['rows']:,} days",
            delta=f"{ui.day(s['start'])} – {ui.day(s['end'])}",
            help=f"{s['targets']:,} prediction targets"
            + (f"; the first {config.TIME_STEP} days serve only as model input." if s["targets"] != s["rows"] else "."),
        )
st.caption(
    "Train and validation periods are listed for reference; everything below covers the test period only. "
    "Filters change what is displayed. They do not retrain or rerun the models."
)

# --- Filters -----------------------------------------------------------------
min_day, max_day = pred["Date"].min().date(), pred["Date"].max().date()
with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
    models = st.pills(
        "Models",
        list(config.MODEL_NAMES),
        selection_mode="multi",
        default=list(config.MODEL_NAMES),
        key="eval_models",
    )
    picked = st.date_input(
        "Test window",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        format="YYYY-MM-DD",
        key="eval_window",
        help="Limit the chart, the window metrics and the download to these dates.",
        width=280,
    )

# While a range is being picked the widget returns a single date; keep the full period until both ends are set.
window = (picked[0], picked[1]) if isinstance(picked, tuple) and len(picked) == 2 else (min_day, max_day)
models = [m for m in config.MODEL_NAMES if m in (models or [])]
view = pred[(pred["Date"].dt.date >= window[0]) & (pred["Date"].dt.date <= window[1])].reset_index(drop=True)
is_full = window == (min_day, max_day)

if view.empty:
    st.info("No test days fall inside the selected window. Widen the date range.")
    st.stop()

# --- Chart -------------------------------------------------------------------
with ui.card("avp"):
    st.subheader("Actual vs predicted close", anchor=False)
    st.plotly_chart(
        charts.actual_vs_predicted(view, models),
        config=charts.PLOTLY_CONFIG,
        alt="Actual crude oil closing prices compared with LSTM and Simple RNN one-step-ahead predictions over the test period.",
        key="avp_chart",
    )
    if not models:
        st.caption("Select at least one model to show its predictions.")
    else:
        st.caption(f"{len(view):,} test days shown, {ui.day(view['Date'].iloc[0])} – {ui.day(view['Date'].iloc[-1])}.")

# --- Metrics -----------------------------------------------------------------
order = [*models, "Persistence baseline"]
full_metrics = meta["test_metrics"]
window_metrics = {name: pipeline.regression_metrics(view["Actual"], view[name]) for name in order}

left, right = st.columns([3, 2], gap="medium")
with left:
    with ui.card("metrics"):
        st.subheader("Error metrics", anchor=False)
        st.markdown("**Full test period**")
        st.dataframe(
            ui.metrics_frame(full_metrics, order),
            hide_index=True,
            column_config=ui.METRIC_COLUMN_CONFIG,
            alt="RMSE, MAE and R squared for each model over the full test period.",
        )
        if is_full:
            st.caption("The selected window covers the full test period.")
        else:
            st.markdown(f"**Selected window** ({len(view):,} days)")
            st.dataframe(
                ui.metrics_frame(window_metrics, order),
                hide_index=True,
                column_config=ui.METRIC_COLUMN_CONFIG,
                alt="RMSE, MAE and R squared for each model within the selected window.",
            )
            st.caption("R² over a short window can be low or negative even when errors are small, because it compares errors with the window's own price variance.")

with right:
    with ui.card("reading"):
        st.subheader("How to read these numbers", anchor=False)
        st.markdown(
            "- **MAE**: average size of a miss, in dollars per barrel.\n"
            "- **RMSE**: like MAE but weights large misses more heavily.\n"
            "- **R²**: share of price variation explained, where 1.0 is perfect. It is not an accuracy percentage.\n"
            "- **Persistence baseline**: predicts that the next close equals the latest close. A useful model should beat it."
        )

# --- Comparison --------------------------------------------------------------
scope = full_metrics if is_full else window_metrics
scope_label = "over the full test period" if is_full else "in the selected window"
with ui.card("compare"):
    st.subheader("Comparison", anchor=False)
    if len(models) == 2:
        lstm, rnn = scope["LSTM"]["RMSE"], scope["Simple RNN"]["RMSE"]
        better, worse = ("LSTM", "Simple RNN") if lstm <= rnn else ("Simple RNN", "LSTM")
        gap = abs(lstm - rnn) / max(lstm, rnn) * 100
        if gap < config.NEGLIGIBLE_GAP_PCT:
            st.markdown(
                f"LSTM and Simple RNN perform almost identically {scope_label}: RMSE {ui.usd(lstm)} and {ui.usd(rnn)}, "
                f"a {gap:.1f}% difference."
            )
        else:
            st.markdown(
                f"{better} has the lower RMSE {scope_label}: {ui.usd(scope[better]['RMSE'])} against "
                f"{ui.usd(scope[worse]['RMSE'])} for {worse}, about {gap:.0f}% lower."
            )
    base = scope["Persistence baseline"]["RMSE"]
    for name in models:
        diff = (scope[name]["RMSE"] - base) / base * 100
        relation = "higher" if diff > 0 else "lower"
        st.markdown(f"{name} RMSE is {abs(diff):.0f}% {relation} than the persistence baseline ({ui.usd(base)}) {scope_label}.")
    st.caption(
        "Day-to-day oil prices are close to a random walk, so a naive \"same as yesterday\" forecast is hard to beat on one-step error. "
        "A high R² mainly reflects that tomorrow's price is close to today's."
    )

with ui.card("errors"):
    st.subheader("Prediction error over time", anchor=False)
    if models:
        st.plotly_chart(
            charts.prediction_errors(view, models),
            config=charts.PLOTLY_CONFIG,
            alt="Daily prediction error, predicted minus actual, for each selected model.",
            key="err_chart",
        )
        st.caption("Positive values mean the model predicted above the actual close.")
    else:
        st.caption("Select at least one model to show its errors.")

# --- Download ----------------------------------------------------------------
export = view[["Date", "Actual", *models]].copy()
for name in models:
    export[f"{name} error"] = export[name] - export["Actual"]
export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
st.download_button(
    "Download displayed results (CSV)",
    export.to_csv(index=False).encode("utf-8"),
    file_name=f"oil_test_predictions_{window[0]:%Y%m%d}_{window[1]:%Y%m%d}.csv",
    mime="text/csv",
    icon=":material/download:",
    on_click="ignore",
    key="eval_download",
)

# --- Provenance --------------------------------------------------------------
st.space("small")
with ui.card("provenance"):
    st.subheader("Where these results come from", anchor=False)
    training = meta["training"]["models"]
    intro = (
        "The models and scaler come from the Google Colab training run recorded in the project notebook. "
        "The app reloads them and recomputes every prediction on this page; nothing is retrained."
    )
    if art.comparison is not None:
        gap = max(
            abs(full_metrics[name][k] - float(art.comparison.loc[name, k]))
            for name in config.MODEL_NAMES
            if name in art.comparison.index
            for k in ("MAE", "RMSE", "R2")
        )
        intro += f" The recomputed metrics match the notebook's saved comparison table to within {gap:.1e}."
    st.markdown(intro)

    if art.history is not None:
        st.markdown("**Training history (from the notebook log)**")
        loss_model = st.segmented_control(
            "Model", list(config.MODEL_NAMES), default="LSTM", required=True, key="loss_model", label_visibility="collapsed"
        )
        info = training[loss_model]
        st.plotly_chart(
            charts.training_loss(art.history, loss_model, weights_epoch=info["weights_epoch"]),
            config=charts.PLOTLY_CONFIG,
            alt=f"Training and validation loss per epoch for the {loss_model} Colab run.",
            key="loss_chart",
        )
        st.caption(
            f"{loss_model}: {info['epochs_run']} epochs with its own early stopping; the saved model holds the weights from "
            f"epoch {info['weights_epoch']}, its lowest validation loss. Loss is mean squared error on scaled prices, as printed by Keras in the notebook."
        )
