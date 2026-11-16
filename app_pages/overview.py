import pandas as pd
import streamlit as st

from oil_dashboard import charts, config, pipeline, store, ui

df = store.prices()
bounds = pipeline.split_bounds(len(df))
art = store.artifacts()
first, last = df["Date"].iloc[0], df["Date"].iloc[-1]

st.markdown(
    "Two recurrent neural networks, an LSTM and a Simple RNN, learn from the previous "
    f"{config.TIME_STEP} daily closing prices to predict the next close. "
    "Both use the same data, preprocessing and test period, so their errors can be compared directly."
)

c1, c2, c3, c4 = ui.kpi_row(4)
with c1:
    ui.kpi("coverage", "Data coverage", f"{first.year}–{last.year}", help=f"{ui.day(first)} to {ui.day(last)}")
with c2:
    ui.kpi("rows", "Daily observations", f"{len(df):,}", help="Trading days in the dataset")
with c3:
    ui.kpi(
        "last_close",
        "Last close in dataset",
        ui.usd(float(df["Close"].iloc[-1])),
        help="Historical value from the dataset, not a live market price.",
    )
with c4:
    ui.kpi("last_date", "Last observation", ui.day(last), help="The dataset is static; it is not updated automatically.")

st.caption(
    "Crude oil futures closing prices in US dollars per barrel (Kaggle crude oil dataset, 2000–2024). "
    "The contract is not named in the file; the negative close on 20 Apr 2020 is consistent with WTI front-month futures."
)

st.space("small")
with ui.card("history"):
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.subheader("Closing price history", anchor=False, width="content")
        span = st.segmented_control(
            "Time range",
            ["1Y", "3Y", "5Y", "10Y", "All"],
            default="All",
            required=True,
            key="overview_range",
            label_visibility="collapsed",
        )
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(span)
    start = max(first, last - pd.DateOffset(years=years)) if years else first
    st.plotly_chart(
        charts.price_history(df, bounds, start, last),
        config=charts.PLOTLY_CONFIG,
        alt="Daily crude oil closing price, with the model's train, validation and test periods shaded.",
        key="history_chart",
    )
    st.caption(
        f"Shaded periods show the chronological split used for modeling: train {ui.day(df['Date'].iloc[0])} – "
        f"{ui.day(df['Date'].iloc[bounds.train_end - 1])}, validation {ui.day(df['Date'].iloc[bounds.train_end])} – "
        f"{ui.day(df['Date'].iloc[bounds.val_end - 1])}, test {ui.day(df['Date'].iloc[bounds.val_end])} – {ui.day(last)}."
    )

st.space("small")
with ui.card("performance"):
    st.subheader("Model performance on the test period", anchor=False)
    if art.metadata is None:
        st.warning(
            f"Run `{store.SETUP_COMMAND}` once to create the trained models and saved test predictions.",
            title="Saved results not found",
        )
    else:
        metrics = art.metadata["test_metrics"]
        test = art.metadata["splits"]["test"]
        rmse = {m: metrics[m]["RMSE"] for m in config.MODEL_NAMES}
        gap_pct = (max(rmse.values()) - min(rmse.values())) / max(rmse.values()) * 100
        best = min(rmse, key=lambda m: rmse[m]) if gap_pct >= config.NEGLIGIBLE_GAP_PCT else None
        cols = st.columns(len(config.MODEL_NAMES), gap="medium")
        for col, name in zip(cols, config.MODEL_NAMES):
            m = metrics[name]
            with col:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**{name}**")
                    if name == best:
                        st.badge("Lower error", color="green", icon=":material/check:")
                with st.container(horizontal=True):
                    st.metric("RMSE", ui.usd(m["RMSE"]), help="Root mean squared error, USD per barrel")
                    st.metric("MAE", ui.usd(m["MAE"]), help="Mean absolute error, USD per barrel")
                    st.metric("R²", f"{m['R2']:.3f}", help="Share of variance explained; not an accuracy percentage")
        baseline = metrics["Persistence baseline"]["RMSE"]
        other = next((m for m in config.MODEL_NAMES if m != best), None)
        verdict = (
            f"In this run the two models' RMSE differs by {gap_pct:.1f}%, which is negligible."
            if best is None
            else f"In this run, {best} has about {gap_pct:.0f}% lower RMSE than the {other}."
        )
        beats = [m for m in config.MODEL_NAMES if rmse[m] < baseline]
        baseline_note = (
            "better than both models" if not beats else "worse than both models" if len(beats) == len(rmse) else f"better than the {other}"
        )
        st.markdown(
            f"{verdict} A naive forecast that repeats the latest close scores an RMSE of {ui.usd(baseline)} on the same days, "
            f"{baseline_note}."
        )
        st.caption(
            f"One-step-ahead predictions on {test['targets']:,} test days ({ui.day(test['start'])} – {ui.day(test['end'])}), "
            "recomputed from the models saved by the Google Colab run in the project notebook."
        )
