"""Small shared UI pieces: header, card styling and number formatting."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from oil_dashboard import config

C = config.COLORS

# Only public hooks are styled: containers created with key="card_*" receive
# the generated `.st-key-card_*` class. Everything else uses the theme config.
_CSS = f"""
<style>
body {{ font-variant-numeric: tabular-nums; }}
[class*="st-key-card"] {{
  background: {C["surface"]};
  border-color: {C["border"]};
  box-shadow: 0 1px 2px rgba(18, 48, 71, 0.05), 0 6px 16px rgba(18, 48, 71, 0.04);
}}
/* Header illustration: scales with its container; smaller on tablets, hidden on phones. */
.st-key-header_art {{ width: 180px; }}
.st-key-header_art img {{ display: block; width: 100%; height: auto; }}
@media (max-width: 900px) {{ .st-key-header_art {{ width: 130px !important; }} }}
@media (max-width: 640px) {{ .st-key-header_art {{ display: none; }} }}
</style>
"""


def inject_styles() -> None:
    st.html(_CSS)


BARRELS_SVG = config.ROOT / "assets" / "oil_barrels.svg"


@st.cache_data(show_spinner=False)
def _decorative_image(path: str, mtime: float) -> str:
    """The local SVG as an <img> data URI (st.html strips inline <svg>). Empty alt: it is decorative."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="" width="260" height="182">'


def app_header() -> None:
    """Shared page header: title on the left, barrel illustration on the right."""
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute", gap="medium"):
        with st.container(width="stretch"):
            st.title("Oil Price Prediction", anchor=False)
            st.subheader(":gray[LSTM & RNN Forecasting Dashboard]", anchor=False)
        if BARRELS_SVG.exists():
            with st.container(width="content", key="header_art"):
                st.html(_decorative_image(str(BARRELS_SVG), BARRELS_SVG.stat().st_mtime))


def footer() -> None:
    st.space("medium")
    st.caption(
        "Personal project showcase by Yasir Aladwani. Not an official Saudi Aramco product or endorsement. "
        "Prices come from a historical dataset, not a live market feed."
    )


def card(key: str, **kwargs):
    """A white bordered container (see `_CSS`)."""
    return st.container(border=True, key=f"card_{key}", **kwargs)


def kpi(key: str, label: str, value: str, help: str | None = None, delta: str | None = None) -> None:
    """A metric in its own white card. Deltas here are neutral context, not good/bad signals."""
    with card(f"kpi_{key}", height="stretch"):
        st.metric(label, value, delta=delta, delta_color="off", delta_arrow="off", help=help)


def kpi_row(n: int):
    """Equal-width columns for KPI cards; Streamlit stacks them on narrow screens."""
    return st.columns(n, gap="medium")


def usd(value: float, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    sign = "+" if signed and value > 0 else ("−" if value < 0 else "")
    return f"{sign}${abs(value):,.2f}"


def pct(value: float) -> str:
    """Signed percent with an ASCII minus, so st.metric deltas pick the right arrow."""
    return f"{value:+.2f}%"


def day(ts: pd.Timestamp | str) -> str:
    """Unambiguous date format, e.g. 30 Oct 2024."""
    return pd.Timestamp(ts).strftime("%d %b %Y").lstrip("0")


def metrics_frame(metrics: dict[str, dict[str, float]], order: list[str]) -> pd.DataFrame:
    rows = [{"Model": name, **metrics[name]} for name in order if name in metrics]
    return pd.DataFrame(rows).rename(columns={"R2": "R²"})


METRIC_COLUMN_CONFIG = {
    "MAE": st.column_config.NumberColumn("MAE (USD)", format="%.3f", help="Mean absolute error: average size of a miss, in USD per barrel."),
    "RMSE": st.column_config.NumberColumn("RMSE (USD)", format="%.3f", help="Root mean squared error: like MAE but penalizes large misses more."),
    "R²": st.column_config.NumberColumn("R²", format="%.4f", help="Share of price variance explained (1.0 is perfect). Not an accuracy percentage."),
}
