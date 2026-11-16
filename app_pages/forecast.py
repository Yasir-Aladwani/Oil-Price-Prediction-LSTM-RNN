from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from oil_dashboard import charts, config, pipeline, store, ui

df = store.prices()
art = store.artifacts()
data_sha = store.data_sha256()

st.header("Interactive forecast", anchor=False)
st.markdown(
    f"Runs a trained model on the latest {config.TIME_STEP} observations in the dataset and predicts the closing price "
    "of the next observation. The models support one-step-ahead forecasts only."
)

window = pipeline.latest_window(df)
last_date = window["Date"].iloc[-1]
last_close = float(window["Close"].iloc[-1])

# --- Inputs ------------------------------------------------------------------
with ui.card("inputs"):
    with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
        model_name = st.segmented_control(
            "Model",
            list(config.MODEL_NAMES),
            default="LSTM",
            required=True,
            key="fc_model",
        )
        missing = art.missing_for_inference(model_name)
        clicked = st.button(
            "Generate Forecast",
            type="primary",
            icon=":material/online_prediction:",
            disabled=bool(missing),
            key="fc_generate",
        )

    c1, c2, c3, c4 = ui.kpi_row(4)
    c1.metric("Last observation", ui.day(last_date), border=True, height="stretch")
    c2.metric("Last close", ui.usd(last_close), border=True, height="stretch", help="Historical dataset value, not a live price.")
    c3.metric(
        "Observations used",
        f"{len(window)}",
        border=True,
        height="stretch",
        help=f"Input window {ui.day(window['Date'].iloc[0])} – {ui.day(last_date)}",
    )
    c4.metric("Forecast target", "Next observation", border=True, height="stretch", help="The exact calendar date is not assumed.")
    st.caption(
        f"Input window: {ui.day(window['Date'].iloc[0])} – {ui.day(last_date)}. The prices are scaled with the scaler fitted "
        f"on the training period, passed through the {model_name} model, and converted back to US dollars per barrel. "
        f"The target is the next trading session after {ui.day(last_date)}; the dataset has no future calendar."
    )

if missing:
    st.warning(
        "Missing: " + "; ".join(missing) + f". Run `{store.SETUP_COMMAND}` once from the project folder "
        "(and install the packages in `requirements.txt`). Saved test results remain available on the Historical evaluation page.",
        title="Forecasting with this model is unavailable",
    )
    st.stop()

if art.data_matches(data_sha) is False:
    st.warning(
        "`data/Crude_Oil_Data.csv` differs from the file the models and scaler were built from. "
        "Forecasts still use the saved training scaler without refitting, but the results may not be comparable.",
        title="Dataset changed since training",
    )

fingerprint = {
    "model": model_name,
    "data_sha": data_sha,
    "artifacts": store.artifact_fingerprint(art, model_name),
    "window_end": last_date.isoformat(),
}

# --- Run ---------------------------------------------------------------------
if clicked:
    assert art.scaler_path is not None  # guaranteed by the `missing` check above
    try:
        with st.spinner(f"Running the {model_name} model...", show_time=True):
            model = store.load_model(art.model_paths[model_name])
            scaler = store.load_scaler(art.scaler_path)
            prediction = pipeline.predict_next(model, scaler, window["Close"].to_numpy())
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}", title="The forecast could not be generated")
        st.stop()
    st.session_state.forecasts[model_name] = {
        **fingerprint,
        "prediction": prediction,
        "last_close": last_close,
        "window_start": window["Date"].iloc[0].isoformat(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

# --- Result ------------------------------------------------------------------
stored = st.session_state.forecasts.get(model_name)
if stored is not None and any(stored[k] != v for k, v in fingerprint.items()):
    # Built from other data or model files: never present it as current.
    del st.session_state.forecasts[model_name]
    st.info(
        f"The data or model files changed since the last {model_name} forecast, so it was cleared. "
        "Select **Generate Forecast** to run the model again.",
        icon=":material/refresh:",
    )
    st.stop()

if stored is None:
    st.info(f"No forecast yet for {model_name}. Select **Generate Forecast** to run the model.", icon=":material/online_prediction:")
    st.stop()

prediction = stored["prediction"]
change = prediction - last_close
with ui.card("result"):
    st.subheader(f"{model_name} forecast for the next observation", anchor=False)
    r1, r2, r3 = ui.kpi_row(3)
    r1.metric("Forecast close", ui.usd(prediction), border=True, height="stretch")
    r2.metric(
        "Change from last close",
        ui.usd(change, signed=True),
        delta=ui.pct(change / last_close * 100),
        delta_color="off",
        border=True,
        height="stretch",
    )
    r3.metric("Based on close of", ui.day(last_date), border=True, height="stretch")

    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.markdown("**Recent closes and forecast**", width="content")
        context_len = st.segmented_control(
            "History shown",
            [60, 120, 250],
            format_func=lambda n: f"{n} days",
            default=120,
            required=True,
            key="fc_context",
            label_visibility="collapsed",
        )
    context = df[["Date", "Close"]].iloc[-context_len:].reset_index(drop=True)
    st.plotly_chart(
        charts.forecast_context(context, prediction, model_name, input_start=window["Date"].iloc[0]),
        config=charts.PLOTLY_CONFIG,
        alt=f"Recent closing prices and the {model_name} forecast for the next observation.",
        key="fc_chart",
    )
    st.caption(
        f"The shaded band marks the {config.TIME_STEP} observations the model used. The forecast is a single point estimate; "
        "no confidence interval was computed. It is placed one step after the last observation for display only. "
        f"Generated {stored['generated_at']}."
    )

    export = pd.DataFrame(
        [
            {
                "model": model_name,
                "target": "next observation after last_observation_date",
                "forecast_close_usd": round(prediction, 4),
                "last_close_usd": round(last_close, 4),
                "change_usd": round(change, 4),
                "change_pct": round(change / last_close * 100, 4),
                "last_observation_date": last_date.strftime("%Y-%m-%d"),
                "input_window_start": pd.Timestamp(stored["window_start"]).strftime("%Y-%m-%d"),
                "observations_used": config.TIME_STEP,
                "dataset_sha256": data_sha,
                "generated_at": stored["generated_at"],
            }
        ]
    )
    st.download_button(
        "Download forecast (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"oil_forecast_{model_name.lower().replace(' ', '_')}_{last_date:%Y%m%d}.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
        key="fc_download",
    )
