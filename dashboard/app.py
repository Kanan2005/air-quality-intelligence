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
sys.path.append(str(Path(__file__).resolve().parent.parent / "hyperlocal_forecast"))
import asyncio
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from govt_aqi_client import GovernmentAQIClient, INDIAN_CITIES
from aqi_forecast import AQIForecaster
from source_attribution_forecasting import attribute_station
from openmeteo_client import fetch_hyperlocal_forecast
from groq_advisory import generate_multilingual_advisory
from aqi_service import fetch_aqi
from weather_service import fetch_weather
from geospatial_service import fetch_geospatial_features
from fire_service import fetch_fire_data
from source_attribution import get_model
from feature_engineering import build_feature_vector, compute_data_quality_flags
from explainability import build_explanation, compute_confidence_score

from mock_data import india_aqi_from_pollutants, _pm25_to_aqi, _pm10_to_aqi, _aqi_category

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

tab_forecasting, tab_geospatial, tab_hyperlocal, tab_insights, tab_settings = st.tabs([
    "📊 ForeCasting",
    "🗺️ Geospatial Source Attribution",
    "🔮 Hyperlocal Forecasting",
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
# ========== GEOSPATIAL TAB ==========
with tab_geospatial:
    st.header("Geospatial Pollution Source Attribution")
    st.caption("Click anywhere on the map")

    india_map = folium.Map(location=[22.5, 80], zoom_start=5, tiles="CartoDB positron")
    map_state = st_folium(india_map, width=None, height=480, key="geo_map")

    st.session_state.setdefault("geo_lat", 28.6139)
    st.session_state.setdefault("geo_lon", 77.2090)

    if map_state and map_state.get("last_clicked"):
        st.session_state["geo_lat"] = map_state["last_clicked"]["lat"]
        st.session_state["geo_lon"] = map_state["last_clicked"]["lng"]

    col_lat, col_lon, col_radius = st.columns(3)
    lat = col_lat.number_input("Latitude", format="%.6f", key="geo_lat")
    lon = col_lon.number_input("Longitude", format="%.6f", key="geo_lon")
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


# ========== HYPERLOCAL FORECAST TAB ==========
with tab_hyperlocal:
    st.header("🔮 Hyperlocal AQI Forecasting")
    st.caption("Click anywhere in the world")

    india_map_h = folium.Map(location=[22.5, 80], zoom_start=5, tiles="CartoDB positron")
    map_state_h = st_folium(india_map_h, width=None, height=460, key="hyperlocal_map")

    st.session_state.setdefault("hyperlocal_lat", 28.6139)
    st.session_state.setdefault("hyperlocal_lon", 77.2090)

    if map_state_h and map_state_h.get("last_clicked"):
        st.session_state["hyperlocal_lat"] = map_state_h["last_clicked"]["lat"]
        st.session_state["hyperlocal_lon"] = map_state_h["last_clicked"]["lng"]

    col_a, col_b = st.columns(2)
    h_lat = col_a.number_input("Latitude", format="%.6f", key="hyperlocal_lat")
    h_lon = col_b.number_input("Longitude", format="%.6f", key="hyperlocal_lon")

    languages = st.multiselect(
        "Advisory languages",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Punjabi"],
        default=["English", "Hindi"],
    )

    if st.button("Get Forecast", key="hyperlocal_btn"):
        hourly = asyncio.run(fetch_hyperlocal_forecast(h_lat, h_lon))

        n_hours = len(hourly["time"])
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "pm2_5": hourly["pm2_5"],
            "pm10": hourly["pm10"],
            "us_aqi": hourly.get("us_aqi", [None] * n_hours),
        })

        df["pm2_5"] = df["pm2_5"].interpolate(limit=3, limit_direction="both")
        df["pm10"] = df["pm10"].interpolate(limit=3, limit_direction="both")

        df["india_aqi_raw"] = df.apply(
            lambda r: india_aqi_from_pollutants(r["pm2_5"], r["pm10"]), axis=1
        )

        # --- Live-station calibration -----------------------------------
        # Open-Meteo's air quality forecast only has high-resolution (11km)
        # coverage over Europe. Everywhere else, including India, falls
        # back to the coarse 45km global CAMS model, which is known to
        # over-predict dust-driven PM10 in arid/desert-adjacent regions
        # (e.g. Rajasthan) since it isn't locally validated against Indian
        # ground stations. Where a live CPCB station is available nearby,
        # anchor the forecast to it rather than trusting the raw model.
        calibration_ratio = None
        calibration_note = None
        try:
            anchor = asyncio.run(fetch_aqi(h_lat, h_lon, 25))
            is_live_anchor = "live" in anchor.get("source", "") and not anchor.get("is_distant_fallback")
            current_forecast_aqi = df["india_aqi_raw"].dropna().iloc[0] if df["india_aqi_raw"].notna().any() else None
            if is_live_anchor and current_forecast_aqi and current_forecast_aqi > 0:
                raw_ratio = anchor["aqi"] / current_forecast_aqi
                calibration_ratio = float(min(max(raw_ratio, 0.3), 1.5))
                calibration_note = (
                    f"Calibrated against live CPCB station ({anchor['aqi']} AQI now) "
                )
        except Exception:
            pass  # calibration is best-effort; fall through to raw forecast below

        if calibration_ratio is not None:
            df["india_aqi"] = (df["india_aqi_raw"] * calibration_ratio).round().clip(0, 1000)
        else:
            df["india_aqi"] = df["india_aqi_raw"]

        missing_hours = df["india_aqi"].isna().sum()
        df_valid = df.dropna(subset=["india_aqi"])

        if df_valid.empty:
            st.error("No usable pollutant data was returned for this location — try a different point or check back later.")
        else:
            if calibration_note:
                st.caption(f"📡 {calibration_note}")
            else:
                st.caption(
                    "ℹ️ No live nearby station available to calibrate against — showing the raw global forecast "
                    "model estimate, which can run high for dust-prone regions."
                )
            if missing_hours:
                st.caption(
                    f"⚠️ {missing_hours} of {len(df)} forecast hours had incomplete pollutant data "
                    "(filled from neighboring hours where possible, otherwise excluded)."
                )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_valid["time"], y=df_valid["india_aqi"], name="India AQI", line=dict(color="#d62728")))
            fig.update_layout(height=380, yaxis_title="India AQI", xaxis_title=None)
            st.plotly_chart(fig, width="stretch")

            peak_row = df_valid.loc[df_valid["india_aqi"].idxmax()]
            peak_aqi = int(peak_row["india_aqi"])
            category = _aqi_category(peak_aqi)
            st.metric("Peak forecasted AQI (next 5 days)", peak_aqi, category)
            st.caption(f"Peak expected around {peak_row['time']}")

            # Show which pollutant + concentration actually drove this peak,
            # and how it compares to neighboring hours, so a genuine event
            # can be told apart from a single-hour forecast-model artifact.
            peak_pm25_aqi = _pm25_to_aqi(peak_row["pm2_5"])
            peak_pm10_aqi = _pm10_to_aqi(peak_row["pm10"])
            driver = "PM2.5" if (peak_pm25_aqi or -1) >= (peak_pm10_aqi or -1) else "PM10"
            peak_idx = df_valid.index.get_loc(peak_row.name) if peak_row.name in df_valid.index else None
            neighbor_note = ""
            if peak_idx is not None and 0 < peak_idx < len(df_valid) - 1:
                prev_val = df_valid.iloc[peak_idx - 1]["india_aqi"]
                next_val = df_valid.iloc[peak_idx + 1]["india_aqi"]
                if peak_aqi - max(prev_val, next_val) > 150:
                    neighbor_note = (
                        f" ⚠️ This is {peak_aqi - max(prev_val, next_val):.0f} points above its neighboring hours "
                        f"({prev_val:.0f} → **{peak_aqi}** → {next_val:.0f}) — worth treating as a possible "
                        "forecast-model spike rather than a confirmed event."
                    )
            st.caption(
                f"Driven by **{driver}**: PM2.5={peak_row['pm2_5']:.1f} µg/m³, PM10={peak_row['pm10']:.1f} µg/m³."
                f"{neighbor_note}"
            )

            SEVERE_THRESHOLD = 200
            if peak_aqi >= SEVERE_THRESHOLD:
                st.warning("⚠️ Forecast crosses into unhealthy territory — generating health advisory...")
                dominant_pollutant = "PM2.5" if df_valid["pm2_5"].max() >= df_valid["pm10"].max() / 1.6 else "PM10"
                advisory = asyncio.run(
                    generate_multilingual_advisory(peak_aqi, category, dominant_pollutant, languages or ["English"])
                )
                st.markdown(advisory)
            else:
                st.success("✅ No severe AQI levels forecasted in the next 5 days for this location.")
            
# ========== INSIGHTS TAB ==========
with tab_insights:
    st.header("📈 Air Quality Insights")
    st.info("💡 Coming soon: trend analysis and comparative city statistics.")

# ========== SETTINGS TAB ==========
with tab_settings:
    st.header("⚙️ Settings & Configuration")
    st.info("💡 Coming soon: API key configuration, data refresh rate, notification preferences, and export options.")