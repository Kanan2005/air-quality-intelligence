"""
Urban Air Quality Intelligence — Streamlit Dashboard

Run with:  streamlit run dashboard/app.py

Ties together:
  - ingestion/openaq_client.py  (live government AQI data, falls back to sample data)
  - forecasting/aqi_forecast.py (24-72h hyperlocal forecast + baseline comparison)
  - forecasting/source_attribution.py (heuristic source attribution)
  - geospatial/*                (pollution source-attribution agent)
  - Citizen health advisory panel (rule-based on AQI thresholds; regional
    language strings are stubbed — see docs/architecture.md for the
    upgrade path to full LLM-generated multilingual advisories)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "forecasting"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "geospatial"))

import asyncio
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from govt_aqi_client import GovernmentAQIClient, INDIAN_CITIES
from aqi_forecast import AQIForecaster
from source_attribution_forecasting import attribute_station

from aqi_service import fetch_aqi
from weather_service import fetch_weather
from geospatial_service import fetch_geospatial_features
from fire_service import fetch_fire_data
from source_attribution import get_model
from feature_engineering import build_feature_vector, compute_data_quality_flags
from explainability import build_explanation, compute_confidence_score

OpenAQClient = GovernmentAQIClient

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
    live_df = load_city_data(city)
    has_live_data = (
        not live_df.empty
        and "aqi" in live_df.columns
        and live_df["aqi"].notna().any()
    )
    if has_live_data:
        return live_df

    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv",
        parse_dates=["timestamp"],
    )
    return df[df["city"] == city]


st.title("🌫️ Urban Air Quality Intelligence")
st.caption(
    "Free-data prototype; "
    "Zero-cost, software-only stack"
)

# ---------- Navigation tabs (navbar) — ONE call, ONE set of tabs ----------
tab_forecasting, tab_geospatial, tab_insights, tab_settings = st.tabs([
    "📊 ForeCasting",
    "🗺️ Geospatial Source Attribution",
    "📈 Insights",
    "⚙️ Settings",
])

with st.sidebar:
    st.header("⚙️ Controls")
    city = st.selectbox("City", list(INDIAN_CITIES.keys()), index=0)
    horizon = st.slider("Forecast horizon (hours)", 24, 72, 48, step=24)
    st.markdown("---")
    client = OpenAQClient()
    st.markdown(
        "**Data mode:** " +
        ("🔑 Live Government AQI API" if client.api_key else "📦 Sample data (no API key set)")
    )
    st.markdown(
        "The app now uses the government AQI endpoint with the supplied API key. "
        "Set `GOVT_AQI_API_KEY` to override it."
    )

history = load_history(city)
latest = history.copy()
if "timestamp" in latest.columns:
    latest = latest.sort_values(["station", "timestamp"], ascending=[True, True])
    latest = latest.groupby("station", as_index=False).tail(1)
else:
    latest = latest.drop_duplicates(subset=["station"], keep="first")

# ========== FORECASTING TAB ==========
with tab_forecasting:
    st.header("ForeCasting & Real-Time Monitoring")

    city_avg_aqi = latest["aqi"].mean()
    band_label, band_color = aqi_band(city_avg_aqi)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{city} Current AQI (avg)", f"{city_avg_aqi:.0f}", band_label)
    col2.metric("Stations Reporting", len(latest))
    col3.metric("Worst Station AQI", f"{latest['aqi'].max():.0f}")
    col4.metric("Best Station AQI", f"{latest['aqi'].min():.0f}")

    st.markdown(f"**Health Advisory:** {health_advisory(city_avg_aqi)}")

    st.divider()

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
        if len(history["station"].unique()) == 0:
            st.info("No station data available for this city.")
        else:
            station_choice = st.selectbox("Station", sorted(history["station"].unique()))
            station_hist = history[history["station"] == station_choice].sort_values("timestamp") if "timestamp" in history.columns else history[history["station"] == station_choice]

            fig = go.Figure()
            recent = station_hist.tail(72)
            if not recent.empty and "aqi" in recent.columns and recent["aqi"].notna().any():
                fig.add_trace(go.Scatter(x=recent["timestamp"] if "timestamp" in recent.columns else range(len(recent)), y=recent["aqi"], name="Observed", line=dict(color="#1f77b4")))

            if len(station_hist) > 24:
                forecaster = AQIForecaster().fit(station_hist)
                forecast_df = forecaster.forecast_station(station_hist, horizon_hours=horizon)
                if not forecast_df.empty:
                    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
                                             name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
                if forecaster.mae_ is not None and forecaster.baseline_mae_ not in {None, 0}:
                    caption_text = (
                        f"Model MAE: **{forecaster.mae_:.1f}** vs Persistence baseline MAE: "
                        f"**{forecaster.baseline_mae_:.1f}** "
                        f"({(1 - forecaster.mae_/forecaster.baseline_mae_)*100:.0f}% improvement)"
                    )
                else:
                    caption_text = "Forecasting is using a simple fallback because the available history is too short for model training."
            else:
                caption_text = "⚠️ Current data is a snapshot without historical time series. Forecast model requires at least 24 hours of history. Using sample data would enable forecasting."

            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="AQI", xaxis_title=None, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption_text)

    st.divider()

    if not latest.empty:
        st.subheader("🚨 Enforcement Priority Queue")
        worst = latest.sort_values("aqi", ascending=False).head(10)
        if not worst.empty:
            priority_text = ""
            for idx, (_, row) in enumerate(worst.iterrows(), 1):
                label, color = aqi_band(row["aqi"])
                priority_text += f"**{idx}. {row['station']}** — AQI {row['aqi']:.0f} ({label})\n\n"
            st.markdown(priority_text)
            st.caption("Ranked by current AQI. Production version would weight by population density + vulnerable site proximity.")

    st.divider()
    st.caption(
        "⚠️ The dashboard now displays the latest API snapshot when available and falls back to sample data only if the API returns no usable rows. "
        "see docs/architecture.md for the real-data upgrade path."
    )

# ========== GEOSPATIAL TAB ==========
with tab_geospatial:
    st.header("Geospatial Pollution Source Attribution")
    st.caption("Click anywhere on the map")

    india_map = folium.Map(location=[22.5, 80], zoom_start=5, tiles="CartoDB positron")
    map_state = st_folium(india_map, width=None, height=480, key="geo_map")

    lat, lon = 28.6139, 77.2090
    if map_state and map_state.get("last_clicked"):
        lat, lon = map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"]

    col_lat, col_lon, col_radius = st.columns(3)
    lat = col_lat.number_input("Latitude", value=lat, format="%.6f")
    lon = col_lon.number_input("Longitude", value=lon, format="%.6f")
    radius_km = col_radius.slider("Search radius (km)", 1, 50, 3)

    if st.button("Analyze Pollution Sources"):
        aqi_data = asyncio.run(fetch_aqi(lat, lon, radius_km))
        weather_data = asyncio.run(fetch_weather(lat, lon))
        geo_data = asyncio.run(fetch_geospatial_features(lat, lon, radius_km))
        fire_data = asyncio.run(fetch_fire_data(lat, lon, radius_km))

        if aqi_data.get("is_distant_fallback"):
            st.warning(
                f"No monitored station within {radius_km} km. Showing data from the "
                f"nearest available CPCB station, {aqi_data['station_distance_km']} km away."
            )

        feature_vector = build_feature_vector(aqi_data, weather_data, geo_data, fire_data)
        source_contribution, tree_agreement = get_model().predict(feature_vector)
        flags = compute_data_quality_flags(aqi_data, weather_data, geo_data, fire_data)
        confidence = compute_confidence_score(tree_agreement, flags, aqi_data)
        explanation, dominant_source, recommendations = build_explanation(
            source_contribution, aqi_data, weather_data, geo_data, fire_data
        )

        st.metric("AQI", aqi_data["aqi"], aqi_data["category"])
        is_live = "live" in aqi_data.get("source", "")
        if is_live:
            distance_note = ""
            if aqi_data.get("station_distance_km") is not None:
                distance_note = f" · nearest station {aqi_data['station_distance_km']} km away"
                if aqi_data.get("is_distant_fallback"):
                    distance_note += " (beyond your search radius)"
            st.success(f"✅ Live CPCB station data{distance_note}")
        else:
            st.warning("⚠️ Synthetic fallback data — CPCB station data was unavailable for this point.")
        st.bar_chart(source_contribution)
        st.markdown(f"**Dominant source:** {dominant_source} · **Confidence:** {confidence:.1f}%")
        st.info(explanation)
        st.write(recommendations)
        st.divider()
        st.subheader("📡 Context Data")

        ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

        with ctx_col1:
            st.markdown("**🌤️ Weather**")
            st.write(f"Temperature: {weather_data['temperature_c']}°C")
            st.write(f"Humidity: {weather_data['humidity_pct']}%")
            st.write(f"Wind: {weather_data['wind_speed_mps']} m/s @ {weather_data['wind_direction_deg']}°")
            st.write(f"Pressure: {weather_data['pressure_hpa']} hPa")
            st.caption(f"Source: {weather_data['source']}")

        with ctx_col2:
            st.markdown("**🛣️ Land Use / Geospatial**")
            st.write(f"Road density: {geo_data['road_density_km_per_km2']} km/km²")
            st.write(f"Major roads nearby: {geo_data['major_road_count']}")
            st.write(f"Industrial land: {geo_data['industrial_area_ratio']*100:.1f}%")
            st.write(f"Construction sites: {int(geo_data['construction_site_count'])}")
            st.write(f"Green cover: {geo_data['green_cover_ratio']*100:.1f}%")
            st.caption(f"Source: {geo_data['source']}")

        with ctx_col3:
            st.markdown("**🔥 Fire / Biomass Burning**")
            st.write(f"Active fire hotspots: {fire_data['active_fire_count']}")
            if fire_data.get("nearest_fire_distance_km") is not None:
                st.write(f"Nearest hotspot: {fire_data['nearest_fire_distance_km']} km away")
            else:
                st.write("Nearest hotspot: none detected")
            if fire_data.get("mean_frp_mw") is not None:
                st.write(f"Mean fire intensity: {fire_data['mean_frp_mw']} MW")
            st.caption(f"Source: {fire_data['source']}")

# ========== INSIGHTS TAB ==========
with tab_insights:
    st.header("📈 Air Quality Insights")
    st.info("💡 Coming soon: trend analysis and comparative city statistics.")

# ========== SETTINGS TAB ==========
with tab_settings:
    st.header("⚙️ Settings & Configuration")
    st.info("💡 Coming soon: API key configuration, data refresh rate, notification preferences, and export options.")