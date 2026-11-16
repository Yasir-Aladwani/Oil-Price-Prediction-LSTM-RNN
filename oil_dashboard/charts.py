"""Plotly figure builders. Series colors are fixed per entity (never by rank)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from oil_dashboard import config
from oil_dashboard.pipeline import SplitBounds

C = config.COLORS
PRICE_AXIS_TITLE = "USD per barrel"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _base_layout(fig: go.Figure, height: int, *, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font={"family": config.FONT_FAMILY, "color": C["navy"], "size": 13},
        margin={"l": 8, "r": 16, "t": 36 if show_legend else 16, "b": 8},
        hovermode="x unified",
        hoverlabel={"bgcolor": C["surface"], "bordercolor": C["border"], "font": {"color": C["navy"], "size": 13}},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=C["border"],
        ticks="outside",
        tickcolor=C["border"],
        tickfont={"color": C["muted"]},
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor=C["muted"],
    )
    fig.update_yaxes(
        gridcolor=C["grid"],
        zeroline=False,
        tickfont={"color": C["muted"]},
        tickprefix="$",
        tickformat=",.0f",
        title={"text": PRICE_AXIS_TITLE, "font": {"color": C["muted"], "size": 12}},
    )
    return fig


def price_history(df: pd.DataFrame, bounds: SplitBounds, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    """Close price with the notebook's train / validation / test periods shaded."""
    view = df[(df["Date"] >= start) & (df["Date"] <= end)]
    fig = go.Figure(
        go.Scatter(
            x=view["Date"],
            y=view["Close"],
            name="Close",
            mode="lines",
            line={"color": C["navy"], "width": 1.6},
            hovertemplate="$%{y:,.2f}<extra>Close</extra>",
        )
    )
    dates = df["Date"]
    periods = [
        ("Train", dates.iloc[0], dates.iloc[bounds.train_end - 1], None),
        ("Validation", dates.iloc[bounds.train_end], dates.iloc[bounds.val_end - 1], C["val_band"]),
        ("Test", dates.iloc[bounds.val_end], dates.iloc[-1], C["test_band"]),
    ]
    for label, p_start, p_end, color in periods:
        lo, hi = max(p_start, start), min(p_end, end)
        if lo >= hi:
            continue
        if color:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=1, layer="below", line_width=0)
        # Validation reads leftward from its end, Test rightward from its start,
        # so the two labels never collide even when the bands are narrow.
        right = label == "Validation"
        fig.add_annotation(
            x=hi if right else lo,
            xanchor="right" if right else "left",
            xshift=-4 if right else 4,
            y=1.0,
            yref="paper",
            yanchor="bottom",
            text=label,
            showarrow=False,
            font={"size": 12, "color": C["muted"]},
        )
    _base_layout(fig, 430, show_legend=False)
    fig.update_layout(margin={"t": 28})
    return fig


def actual_vs_predicted(view: pd.DataFrame, models: Sequence[str]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=view["Date"],
            y=view["Actual"],
            name="Actual",
            mode="lines",
            line={"color": config.SERIES_COLORS["Actual"], "width": 2},
            hovertemplate="$%{y:,.2f}",
        )
    )
    for name in models:
        fig.add_trace(
            go.Scatter(
                x=view["Date"],
                y=view[name],
                name=f"{name} prediction",
                mode="lines",
                line={"color": config.SERIES_COLORS[name], "width": 1.6},
                hovertemplate="$%{y:,.2f}",
            )
        )
    return _base_layout(fig, 470)


def prediction_errors(view: pd.DataFrame, models: Sequence[str]) -> go.Figure:
    """Signed error (predicted minus actual) per date."""
    fig = go.Figure()
    for name in models:
        fig.add_trace(
            go.Scatter(
                x=view["Date"],
                y=view[name] - view["Actual"],
                name=name,
                mode="lines",
                line={"color": config.SERIES_COLORS[name], "width": 1.3},
                hovertemplate="%{y:+,.2f} USD",
            )
        )
    fig.add_hline(y=0, line={"color": C["muted"], "width": 1})
    _base_layout(fig, 260)
    fig.update_yaxes(tickprefix="", tickformat="+,.0f", title={"text": "Error, USD"})
    return fig


def forecast_context(context: pd.DataFrame, prediction: float, model_name: str, input_start: pd.Timestamp) -> go.Figure:
    """Recent history plus the one-step forecast.

    The forecast marker is drawn one trading day after the last observation for
    readability only; its hover text does not claim a calendar date.
    """
    last_date = context["Date"].iloc[-1]
    last_close = float(context["Close"].iloc[-1])
    marker_x = last_date + pd.offsets.BDay(1)
    fig = go.Figure()
    fig.add_vrect(x0=input_start, x1=last_date, fillcolor=C["val_band"], opacity=1, layer="below", line_width=0)
    fig.add_annotation(
        x=last_date,
        y=1.0,
        yref="paper",
        yanchor="bottom",
        xanchor="right",
        text=f"Model input: last {config.TIME_STEP} observations",
        showarrow=False,
        font={"size": 12, "color": C["muted"]},
    )
    fig.add_trace(
        go.Scatter(
            x=context["Date"],
            y=context["Close"],
            name="Actual close",
            mode="lines",
            line={"color": C["navy"], "width": 2},
            hovertemplate="$%{y:,.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date, marker_x],
            y=[last_close, prediction],
            mode="lines",
            line={"color": config.SERIES_COLORS[model_name], "width": 1.6, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[marker_x],
            y=[prediction],
            name=f"{model_name} forecast for next observation: ${prediction:,.2f}",
            mode="markers",
            marker={"color": config.SERIES_COLORS[model_name], "size": 12, "line": {"color": C["surface"], "width": 2}},
            hovertemplate=f"Next observation<br>{model_name} forecast: $%{{y:,.2f}}<extra></extra>",
        )
    )
    _base_layout(fig, 430)
    # Legend below the plot keeps the top edge free for the input-window label.
    fig.update_layout(
        hovermode="closest",
        margin={"t": 28, "b": 8},
        legend={"yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0},
    )
    # A little room right of the marker, proportional to the history shown.
    pad = pd.offsets.BDay(max(3, round(len(context) * 0.04)))
    fig.update_xaxes(range=[context["Date"].iloc[0], marker_x + pad])
    return fig


def training_loss(history: pd.DataFrame, model_name: str, weights_epoch: int | None = None) -> go.Figure:
    h = history[history["Model"] == model_name]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=h["Epoch"], y=h["Train loss"], name="Train loss", mode="lines", line={"color": config.SERIES_COLORS[model_name], "width": 2})
    )
    fig.add_trace(
        go.Scatter(x=h["Epoch"], y=h["Validation loss"], name="Validation loss", mode="lines", line={"color": C["navy"], "width": 2, "dash": "dash"})
    )
    if weights_epoch is not None and (h["Epoch"] == weights_epoch).any():
        point = h[h["Epoch"] == weights_epoch]
        fig.add_trace(
            go.Scatter(
                x=point["Epoch"],
                y=point["Validation loss"],
                name=f"Saved weights (epoch {weights_epoch})",
                mode="markers",
                marker={"size": 10, "color": C["navy"], "line": {"color": C["surface"], "width": 2}},
                hovertemplate="Saved weights<extra></extra>",
            )
        )
    _base_layout(fig, 280)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title={"text": "Epoch", "font": {"color": C["muted"], "size": 12}}, showspikes=False)
    ticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    fig.update_yaxes(
        type="log",
        tickprefix="",
        tickvals=ticks,
        ticktext=[f"{t:.0e}".replace("e-0", "e-") for t in ticks],
        title={"text": "MSE, scaled (log)"},
    )
    return fig
