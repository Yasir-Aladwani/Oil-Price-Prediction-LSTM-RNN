"""Oil Price Prediction: LSTM & RNN Forecasting Dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st

from oil_dashboard import ui

st.set_page_config(
    page_title="Oil Price Prediction",
    page_icon=":material/oil_barrel:",
    layout="wide",
)

if "forecasts" not in st.session_state:
    # Forecast results keyed by model name; each entry records the exact
    # data and artifact fingerprints it was produced from.
    st.session_state.forecasts = {}

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/space_dashboard:",
            default=True,
            help="Dataset summary, price history with the train, validation and test periods, and a model performance summary.",
        ),
        st.Page(
            "app_pages/evaluation.py",
            title="Historical evaluation",
            icon=":material/insights:",
            url_path="evaluation",
            help="Saved test-period predictions for LSTM and Simple RNN compared with actual prices. Filters change the view only.",
        ),
        st.Page(
            "app_pages/forecast.py",
            title="Interactive forecast",
            icon=":material/online_prediction:",
            url_path="forecast",
            help="Run a trained model on the latest 60 observations to forecast the next observation's close.",
        ),
        st.Page(
            "app_pages/about.py",
            title="About",
            icon=":material/info:",
            url_path="about",
            help="Project goal, data source, model explanations, settings, limitations and author.",
        ),
    ],
    position="top",
)

ui.inject_styles()
ui.app_header()
page.run()
ui.footer()
