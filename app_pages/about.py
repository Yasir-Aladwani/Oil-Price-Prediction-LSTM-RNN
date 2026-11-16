import pandas as pd
import streamlit as st

from oil_dashboard import config, pipeline, store, ui

df = store.prices()
art = store.artifacts()
bounds = pipeline.split_bounds(len(df))
first, last = df["Date"].iloc[0], df["Date"].iloc[-1]
meta = art.metadata or {}
training = meta.get("training", {})
models_info = training.get("models", {})
versions = training.get("library_versions", {})
run_label = ", ".join(
    ["Google Colab", *(f"{lib} {versions[lib]}" for lib in ("TensorFlow", "Keras", "scikit-learn") if lib in versions)]
)

st.header("About the project", anchor=False)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Goal", anchor=False)
    st.markdown(
        "This project tests whether recurrent neural networks can forecast the next daily closing price of crude oil "
        "from recent price history, and compares a Long Short-Term Memory (LSTM) network with a Simple RNN trained "
        "under identical conditions."
    )

    st.subheader("Data", anchor=False)
    st.markdown(
        f"An open historical crude oil dataset from [Kaggle]({config.DATASET_URL}): {len(df):,} daily observations "
        f"from {ui.day(first)} to {ui.day(last)}, with open, high, low, close and volume. The models use only the "
        "**closing price in US dollars per barrel**. The contract is not named in the file; its negative close on "
        "20 Apr 2020 is consistent with WTI front-month futures. The dataset is static, so nothing in this app reflects live prices."
    )

    st.subheader("The two models", anchor=False)
    st.markdown(
        "- **Simple RNN** reads the price sequence one step at a time and carries a single hidden state forward. "
        "It is fast and simple, but the training signal from early steps fades over long sequences "
        "(the vanishing gradient problem).\n"
        "- **LSTM** adds a memory cell controlled by forget, input and output gates, which decide what to keep, update "
        "or pass on. This helps it use information from further back in the 60-day window."
    )

    st.subheader("Workflow", anchor=False)
    st.markdown(
        "1. **Data**: load the CSV, parse dates and sort chronologically.\n"
        "2. **Preprocessing**: split by time into 70% train, 15% validation and 15% test; fit a MinMaxScaler on the "
        "training period only; build sliding windows of 60 closes, each labeled with the following close.\n"
        "3. **Modeling**: train both networks in Google Colab with Adam and mean squared error loss, using early stopping "
        "on validation loss, and save the models and the fitted scaler.\n"
        "4. **Evaluation**: predict each test day from its previous 60 actual closes, convert back to dollars, and "
        "compute MAE, RMSE and R².\n"
        "5. **Inference**: apply the saved scaler and model to the latest 60 observations to forecast the next one."
    )

    st.subheader("Limitations", anchor=False)
    st.markdown(
        "- Forecasts cover **one observation ahead** only. Longer horizons would need recursive or multi-output "
        "methods that this project has not evaluated.\n"
        "- The models see past prices only: no supply, demand, inventory, currency or geopolitical inputs.\n"
        "- A naive \"same as yesterday\" forecast is a strong baseline for daily prices; compare against it on the "
        "Historical evaluation page.\n"
        "- The validation period includes the April 2020 negative price, which lies outside the price range the scaler saw in training.\n"
        "- Results come from a single training run. Another run, even with the same seed on different hardware, can rank the models differently.\n"
        "- No confidence intervals are produced. Results are for learning and demonstration, not trading or investment decisions."
    )

with right:
    with ui.card("settings"):
        st.subheader("Key settings", anchor=False)
        settings = pd.DataFrame(
            [
                ("Target", "Close (USD per barrel)"),
                ("Input window", f"{config.TIME_STEP} observations"),
                ("Forecast horizon", "1 observation ahead"),
                ("Split", "70% / 15% / 15%, chronological"),
                ("Train", f"{ui.day(first)} – {ui.day(df['Date'].iloc[bounds.train_end - 1])}"),
                ("Validation", f"{ui.day(df['Date'].iloc[bounds.train_end])} – {ui.day(df['Date'].iloc[bounds.val_end - 1])}"),
                ("Test", f"{ui.day(df['Date'].iloc[bounds.val_end])} – {ui.day(last)}"),
                ("Scaling", "MinMaxScaler (0–1), fit on train only"),
                ("Architecture", "2 recurrent layers × 50 units, dropout 0.2, dense output"),
                ("Optimizer and loss", "Adam, mean squared error"),
                ("Batch size", "32"),
                ("Epochs", "Up to 50, early stopping per model (patience 8, best weights restored)"),
                ("Training run", run_label),
                ("Random seed", str(training["seed"]) if training.get("seed") is not None else "Not set"),
            ],
            columns=["Setting", "Value"],
        )
        st.table(settings, hide_index=True, hide_header=True, border="horizontal", alt="Data, preprocessing and training settings.")
        if models_info:
            params = ", ".join(f"{name} {info['parameters']:,}" for name, info in models_info.items())
            epochs = ", ".join(
                f"{name} {info['epochs_run']} (best epoch {info['weights_epoch']})" for name, info in models_info.items()
            )
            st.caption(f"Trainable parameters: {params}. Epochs run: {epochs}.")

    with ui.card("author"):
        st.subheader("Author", anchor=False)
        st.markdown("Developed by Yasir Aladwani during his internship at Saudi Aramco.")
        st.caption("Personal project showcase. Not an official Saudi Aramco product or endorsement.")
        st.link_button("View the repository on GitHub", config.REPO_URL, icon=":material/code:")
