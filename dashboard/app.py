"""
Urban Air Quality Intelligence — Streamlit Dashboard

Run with:  streamlit run dashboard/app.py

Ties together:
  - ingestion/openaq_client.py  (live data, falls back to sample data)
  - forecasting/aqi_forecast.py (24-72h hyperlocal forecast + baseline comparison)
  - forecasting/source_attribution.py (heuristic source attribution)
  - Citizen health advisory panel (rule-based on AQI thresholds; regional
    language strings are stubbed — see docs/architecture.md for the
    upgrade path to full LLM-generated multilingual advisories)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "forecasting"))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from openaq_client import OpenAQClient, INDIAN_CITIES
from aqi_forecast import AQIForecaster
from source_attribution import attribute_station

st.set_page_config(page_title="Urban Air Quality Intelligence", layout="wide")

AQI_BANDS = [
    (0, 50, "Good", "#00A651"),
    (51, 100, "Satisfactory", "#A3C853"),
    (101, 200, "Moderate", "#FFD700"),
    (201, 300, "Poor", "#FF7E00"),
    (301, 400, "Very Poor", "#FF0000"),
    (401, 500, "Severe", "#7E0023"),
]


def aqi_band(aqi):
    for lo, hi, label, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color
    return "Severe", "#7E0023"


def health_advisory(aqi):
    if aqi <= 100:
        return "Air quality is acceptable. Outdoor activity is safe for all groups."
    elif aqi <= 200:
        return "Sensitive groups (children, elderly, respiratory/heart conditions) should reduce prolonged outdoor exertion."
    elif aqi <= 300:
        return "Everyone may experience mild effects. Sensitive groups should avoid outdoor exertion; consider masks (N95) outdoors."
    elif aqi <= 400:
        return "Health warning: everyone should limit outdoor exertion. Sensitive groups should stay indoors."
    else:
        return "Health emergency: avoid all outdoor exposure. Keep windows closed; use air purifiers if available."


@st.cache_data(ttl=1800)
def load_city_data(city: str):
    client = OpenAQClient()
    return client.fetch_city_snapshot(city)


@st.cache_data(ttl=1800)
def load_history(city: str):
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv",
        parse_dates=["timestamp"],
    )
    return df[df["city"] == city]


st.title("🌫️ Urban Air Quality Intelligence")
st.caption(
    "Free-data prototype for ET AI Hackathon 2026 · Sources: OpenAQ (CPCB feed), "
    "CAAQMS · Zero-cost, software-only stack"
)

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("City", list(INDIAN_CITIES.keys()), index=0)
    horizon = st.slider("Forecast horizon (hours)", 24, 72, 48, step=24)
    st.markdown("---")
    st.markdown(
        "**Data mode:** " +
        ("🔑 Live OpenAQ API" if OpenAQClient().api_key else "📦 Sample data (no API key set)")
    )
    st.markdown(
        "Set `OPENAQ_API_KEY` env var for live station data. "
        "[Get a free key](https://explore.openaq.org/register)"
    )

history = load_history(city)
latest = history.sort_values("timestamp").groupby("station").tail(1)

# ---------- Top KPI row ----------
city_avg_aqi = latest["aqi"].mean()
band_label, band_color = aqi_band(city_avg_aqi)
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{city} Current AQI (avg)", f"{city_avg_aqi:.0f}", band_label)
col2.metric("Stations Reporting", len(latest))
col3.metric("Worst Station AQI", f"{latest['aqi'].max():.0f}")
col4.metric("Best Station AQI", f"{latest['aqi'].min():.0f}")

st.markdown(f"**Health Advisory:** {health_advisory(city_avg_aqi)}")

st.divider()

# ---------- Map + Forecast side by side ----------
map_col, forecast_col = st.columns([1, 1])

with map_col:
    st.subheader("📍 Station Map (real-time)")
    center_lat, center_lon = INDIAN_CITIES[city]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
    for _, row in latest.iterrows():
        label, color = aqi_band(row["aqi"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=12,
            popup=f"{row['station']}<br>AQI: {row['aqi']:.0f} ({label})",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
        ).add_to(m)
    st_folium(m, width=None, height=420)

with forecast_col:
    st.subheader(f"📈 {horizon}h Forecast — busiest station")
    station_choice = st.selectbox("Station", sorted(history["station"].unique()))
    station_hist = history[history["station"] == station_choice].sort_values("timestamp")

    forecaster = AQIForecaster().fit(history)
    forecast_df = forecaster.forecast_station(station_hist, horizon_hours=horizon)

    fig = go.Figure()
    recent = station_hist.tail(72)
    fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["aqi"], name="Observed", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
                              name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="AQI", xaxis_title=None, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    if forecaster.mae_ is not None:
        st.caption(
            f"Model MAE: **{forecaster.mae_:.1f}** vs Persistence baseline MAE: "
            f"**{forecaster.baseline_mae_:.1f}** "
            f"({(1 - forecaster.mae_/forecaster.baseline_mae_)*100:.0f}% improvement)"
        )

st.divider()

# ---------- Source attribution + Enforcement priority ----------
attr_col, enf_col = st.columns([1, 1])

with attr_col:
    st.subheader("🔍 Source Attribution (heuristic v1)")
    attribution = attribute_station(station_hist)
    for source, conf in attribution["attribution"].items():
        st.progress(conf, text=f"{source.replace('_', ' ').title()}: {conf*100:.0f}%")
    st.caption(attribution["note"])

with enf_col:
    st.subheader("🚨 Enforcement Priority Queue")
    worst = latest.sort_values("aqi", ascending=False).head(5)
    for _, row in worst.iterrows():
        label, color = aqi_band(row["aqi"])
        st.markdown(f"- **{row['station']}** — AQI {row['aqi']:.0f} ({label})")
    st.caption("Ranked by current AQI. Production version would weight by population density + vulnerable site proximity.")

st.divider()
st.caption(
    "⚠️ Prototype disclaimer: dashboard runs on sample data unless OPENAQ_API_KEY is set. "
    "Source attribution is a transparent rule-based proxy, not a trained ML classifier — "
    "see docs/architecture.md for the real-data upgrade path."
)
