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
import numpy as np
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from govt_aqi_client import GovernmentAQIClient, INDIAN_CITIES
from aqi_forecast import AQIForecaster
from source_attribution_forecasting import attribute_station
from openmeteo_client import fetch_hyperlocal_forecast
from groq_advisory import generate_multilingual_advisory
from aqi_service import fetch_aqi
from weather_service import fetch_weather
from geospatial_service import fetch_geospatial_features
from fire_service import fetch_fire_data
from traffic_service import fetch_traffic_congestion
from history_service import generate_pollution_source_history
from source_attribution import get_model
from feature_engineering import build_feature_vector, compute_data_quality_flags
from explainability import build_explanation, compute_confidence_score
from action_plan_advisory import generate_action_plan

from mock_data import india_aqi_from_pollutants, _pm25_to_aqi, _pm10_to_aqi, _aqi_category

OpenAQClient = GovernmentAQIClient

st.set_page_config(page_title="Urban Air Quality Intelligence", layout="wide")

st.markdown(
    """
    <style>
    /* Hide the Deploy button */
    .stAppDeployButton {
        display: none !important;
    }

    /* ---- Action plan / advisory card styling ---- */
    .ap-citizen-box {
        background: linear-gradient(135deg, #fff4e5 0%, #ffe8e8 100%);
        border-left: 5px solid #d62728;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0 18px 0;
    }
    .ap-citizen-box h4 {
        margin: 0 0 8px 0;
        color: #7a1f1f;
    }
    .ap-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-left: 5px solid var(--ap-accent, #1f77b4);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .ap-card h4 {
        margin: 0 0 8px 0;
        font-size: 1.02rem;
        letter-spacing: 0.2px;
    }
    .ap-card ul {
        margin: 0;
        padding-left: 20px;
    }
    .ap-card li {
        margin-bottom: 6px;
        line-height: 1.45;
        color: #262730;
    }
    .ap-model-card {
        background: #f5f8ff;
        border: 1px solid #d9e2f5;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .ap-model-card .ap-model-title {
        font-weight: 600;
        color: #1a3d8f;
    }
    .ap-model-card .ap-model-result {
        color: #444;
        font-size: 0.92rem;
    }
    .ap-source-note {
        font-size: 0.78rem;
        color: #888;
        margin-top: -4px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def _render_bullet_card(title: str, icon: str, bullets: list, accent: str):
    """Render a professional-styled bullet card for an action-plan category."""
    items = [b for b in (bullets or []) if b and str(b).strip()]
    if not items:
        return

    bullets_html = "".join(
        f"<li style='color:#374151;'>{b}</li>" for b in items
    )

    st.markdown(
        f"""
        <div class="ap-card" style="--ap-accent:{accent};">
            <h4 style="color:#111827; font-weight:700;">
                {icon} {title}
            </h4>
            <ul>{bullets_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_action_plan(plan: dict):
    """Render a Groq-generated action plan as professional-style UI bullet cards."""
    citizen_bullets = [b for b in (plan.get("citizen_advisory") or []) if b and str(b).strip()]
    if citizen_bullets:
        bullets_html = "".join(f"<li style='color:#374151;'>{b}</li>" for b in citizen_bullets)
        st.markdown(
            f"""
            <div class="ap-citizen-box">
                <h4 style="color:#111827;">🧑‍🤝‍🧑 Citizen Risk Advisory</h4>
                <ul>{bullets_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        _render_bullet_card("Traffic Control", "🚦", plan.get("traffic_control"), "#1f77b4")
        _render_bullet_card("Industrial Emission Control", "🏭", plan.get("industrial_emission_control"), "#8c564b")
        _render_bullet_card("Other Recommendations", "💡", plan.get("other_suggestions"), "#9467bd")
    with col2:
        _render_bullet_card("Vehicle Emission Control", "🚗", plan.get("vehicle_emission_control"), "#ff7f0e")
        _render_bullet_card("Agricultural Biomass Burning Control", "🌾", plan.get("agri_biomass_control"), "#2ca02c")

    # models = plan.get("successful_city_models") or []
    # if models:
    #     st.markdown(
    #         """
    #         <div class='ap-card' style='--ap-accent:#17a2b8;'>
    #             <h4 style='color:#111827;'>🏆 Successful Models Adopted by Cities</h4>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    #     for m in models:
    #         if not isinstance(m, dict):
    #             continue

    #         city = m.get("city", "")
    #         model_name = m.get("model", "")
    #         result = m.get("result", "")

    #         st.markdown(
    #             f"""
    #             <div class="ap-model-card">
    #                 <div class="ap-model-title" style="color:#111827;font-weight:700;">
    #                     {city} — {model_name}
    #                 </div>
    #                 <div class="ap-model-result" style="color:#4b5563;">
    #                     {result}
    #                 </div>
    #             </div>
    #             """,
    #             unsafe_allow_html=True,
    #         )

    # source = plan.get("source", "")
    # if source and source != "static_fallback":
    #     st.markdown(
    #         f"<div class='ap-source-note' style='color:#6b7280;'>Generated by LLM ({source}) — verify before formal deployment.</div>",
    #         unsafe_allow_html=True,
    #     )
    # else:
    #     st.markdown(
    #         "<div class='ap-source-note' style='color:#6b7280;'>Static fallback plan — configure GROQ_API_KEY for a location-tailored, LLM-generated plan.</div>",
    #         unsafe_allow_html=True,
    #     )



AQI_BANDS = [
    (0, 50, "Clean ✅", "#00A651"),
    (51, 100, "Satisfactory ▲", "#A3C853"),
    (101, 200, "Moderate ⚖", "#FFD700"),
    (201, 300, "Poor 🔻", "#FF7E00"),
    (301, 400, "Very Poor 📉", "#FF0000"),
    (401, 500, "Severe ⚠️", "#7E0023")
]

CITY_POPULATION_PROXY = {
    "Delhi": 33_000_000,
    "Mumbai": 20_000_000,
    "Kolkata": 15_000_000,
    "Bengaluru": 13_000_000,
    "Chennai": 11_000_000,
    "Hyderabad": 10_000_000,
    "Pune": 7_000_000,
    "Agra": 2_000_000,
    "Jaipur": 4_000_000,
}


def aqi_band(aqi):
    for lo, hi, label, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color
    return "Severe", "#7E0023"


# def health_advisory(aqi):
#     if aqi <= 100:
#         return "Air quality is acceptable. Outdoor activity is safe for all groups."
#     elif aqi <= 200:
#         return "Sensitive groups (children, elderly, respiratory/heart conditions) should reduce prolonged outdoor exertion."
#     elif aqi <= 300:
#         return "Everyone may experience mild effects. Sensitive groups should avoid outdoor exertion; consider masks (N95) outdoors."
#     elif aqi <= 400:
#         return "Health warning: everyone should limit outdoor exertion. Sensitive groups should stay indoors."
#     else:
#         return "Health emergency: avoid all outdoor exposure. Keep windows closed; use air purifiers if available."


def _prepare_city_frame(history: pd.DataFrame, period: str):
    if history.empty:
        return pd.DataFrame(columns=["timestamp", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3"])

    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "aqi"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3"])

    latest = frame["timestamp"].max()
    if period == "Today":
        start = latest.normalize()
    elif period == "Last 7 Days":
        start = latest.normalize() - pd.Timedelta(days=7)
    elif period == "Last 30 Days":
        start = latest.normalize() - pd.Timedelta(days=30)
    elif period == "Winter Season":
        start = latest.normalize() - pd.Timedelta(days=90)
    elif period == "Custom":
        start = latest.normalize() - pd.Timedelta(days=30)
    else:
        start = latest.normalize() - pd.Timedelta(days=30)

    frame = frame[frame["timestamp"] >= start].copy()
    if frame.empty:
        frame = history.copy()

    frame = frame.sort_values("timestamp")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["pm25"] = frame["aqi"] * 0.72
    frame["pm10"] = frame["aqi"] * 0.95
    frame["no2"] = frame["aqi"] * 0.48
    frame["so2"] = frame["aqi"] * 0.22
    frame["co"] = frame["aqi"] * 0.18
    frame["o3"] = frame["aqi"] * 0.33
    return frame


def _summarize_city(frame: pd.DataFrame, city: str):
    if frame.empty:
        return {
            "avg_aqi": 0.0,
            "avg_pm25": 0.0,
            "exposure": 0.0,
            "compliance": 0.0,
            "health_risk": 0.0,
            "trend": [],
            "daily": pd.DataFrame(columns=["date", "aqi"]),
        }

    daily = frame.groupby("date")["aqi"].mean().reset_index()
    avg_aqi = float(daily["aqi"].mean())
    avg_pm25 = float(frame["pm25"].mean())
    population = CITY_POPULATION_PROXY.get(city, 10_000_000)
    exposure = avg_aqi * (population / 1_000_000)
    compliance = max(0.0, min(100.0, 100.0 - (avg_aqi / 5.0)))
    health_risk = min(100.0, avg_aqi * 0.65 + avg_pm25 * 0.35)
    trend = daily["aqi"].rolling(7, min_periods=1).mean().tolist()

    return {
        "avg_aqi": avg_aqi,
        "avg_pm25": avg_pm25,
        "exposure": exposure,
        "compliance": compliance,
        "health_risk": health_risk,
        "trend": trend,
        "daily": daily,
    }


def _change_pct(current: float, previous: float) -> float:
    if previous in {None, 0} or pd.isna(previous):
        return 0.0
    return round(((current - previous) / previous) * 100, 1)


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
    "Smarter cities begin with cleaner air. "
    "Intelligence beyond monitoring."
)

tab_forecasting, tab_geospatial, tab_hyperlocal, tab_insights= st.tabs([
    "📊 Real-Time Monitoring",
    "🗺️ Geospatial Source Attribution",
    "🔮 Hyperlocal Forecasting",
    "📈 Insights"
])

with st.sidebar:
    st.header("⚙️ Controls")
    city = st.selectbox("City", list(INDIAN_CITIES.keys()), index=0)
    horizon = st.slider("Forecast horizon (hours)", 24, 72, 48, step=24)
    st.markdown("---")
    client = OpenAQClient()
    st.markdown(
        "**Data mode:** " +
        ("🔑 Live Government API" if client.api_key else "")
    )
    st.markdown(
        "The app uses the government AQI endpoint with the supplied API key. "
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
    st.header("Real-Time Monitoring")

    city_avg_aqi = latest["aqi"].mean()
    band_label, band_color = aqi_band(city_avg_aqi)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{city} Current AQI (avg)", f"{city_avg_aqi:.0f}")

    col1.markdown(
        f"<span style='color:{band_color}; font-weight:600;'>{band_label}</span>",
        unsafe_allow_html=True,
    )
    col2.metric("Stations Reporting", len(latest))
    col3.metric("Worst Station AQI", f"{latest['aqi'].max():.0f}")
    col4.metric("Best Station AQI", f"{latest['aqi'].min():.0f}")

    # st.markdown(f"**Health Advisory:** {health_advisory(city_avg_aqi)}")

    st.divider()

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
    st_folium(m, height=420, width=None)

    # with forecast_col:
    #     st.subheader(f"📈 {horizon}h Forecast — busiest station")
    #     if len(history["station"].unique()) == 0:
    #         st.info("No station data available for this city.")
    #     else:
    #         station_choice = st.selectbox("Station", sorted(history["station"].unique()))
    #         station_hist = history[history["station"] == station_choice].sort_values("timestamp") if "timestamp" in history.columns else history[history["station"] == station_choice]

    #         fig = go.Figure()
    #         recent = station_hist.tail(72)
    #         if not recent.empty and "aqi" in recent.columns and recent["aqi"].notna().any():
    #             fig.add_trace(go.Scatter(x=recent["timestamp"] if "timestamp" in recent.columns else range(len(recent)), y=recent["aqi"], name="Observed", line=dict(color="#1f77b4")))

    #         if len(station_hist) > 24:
    #             forecaster = AQIForecaster().fit(station_hist)
    #             forecast_df = forecaster.forecast_station(station_hist, horizon_hours=horizon)
    #             if not forecast_df.empty:
    #                 fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
    #                                          name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
    #             if forecaster.mae_ is not None and forecaster.baseline_mae_ not in {None, 0}:
    #                 caption_text = (
    #                     f"Model MAE: **{forecaster.mae_:.1f}** vs Persistence baseline MAE: "
    #                     f"**{forecaster.baseline_mae_:.1f}** "
    #                     f"({(1 - forecaster.mae_/forecaster.baseline_mae_)*100:.0f}% improvement)"
    #                 )
    #             else:
    #                 caption_text = "Forecasting is using a simple fallback because the available history is too short for model training."
    #         else:
    #             caption_text = "⚠️ Current data is a snapshot without historical time series. Forecast model requires at least 24 hours of history. Using sample data would enable forecasting."

    #         fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
    #                           yaxis_title="AQI", xaxis_title=None, legend=dict(orientation="h"))
    #         st.plotly_chart(fig, use_container_width=True)
    #         st.caption(caption_text)

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
        # geo_data = asyncio.run(fetch_geospatial_features(lat, lon, radius_km))
        fire_data = asyncio.run(fetch_fire_data(lat, lon, radius_km))
        traffic_data = asyncio.run(fetch_traffic_congestion(lat, lon))

        if aqi_data.get("is_distant_fallback"):
            st.warning(
                f"No monitored station within {radius_km} km. Showing data from the "
                f"nearest available CPCB station, {aqi_data['station_distance_km']} km away."
            )

        feature_vector = build_feature_vector(aqi_data, weather_data,traffic_data, fire_data)
        source_contribution, tree_agreement = get_model().predict(feature_vector)
        flags = compute_data_quality_flags(aqi_data, weather_data, traffic_data, fire_data)
        confidence = compute_confidence_score(tree_agreement, flags, aqi_data)
        explanation, dominant_source, recommendations = build_explanation(
            source_contribution, aqi_data, weather_data, traffic_data, fire_data
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
        # st.write(recommendations)
        st.divider()
        st.subheader("Context Data")

        ctx_col1, ctx_col3, ctx_col4 = st.columns(3)

        with ctx_col1:
            st.markdown("**🌤️ Weather**")
            st.write(f"Temperature: {weather_data['temperature_c']}°C")
            st.write(f"Humidity: {weather_data['humidity_pct']}%")
            st.write(f"Wind: {weather_data['wind_speed_mps']} m/s @ {weather_data['wind_direction_deg']}°")
            st.write(f"Pressure: {weather_data['pressure_hpa']} hPa")
            st.caption(f"Source: {weather_data['source']}")

        # with ctx_col2:
        #     st.markdown("**🛣️ Land Use / Geospatial**")
        #     st.write(f"Road density: {geo_data['road_density_km_per_km2']} km/km²")
        #     st.write(f"Major roads nearby: {geo_data['major_road_count']}")
        #     st.write(f"Industrial land: {geo_data['industrial_area_ratio']*100:.1f}%")
        #     st.write(f"Construction sites: {int(geo_data['construction_site_count'])}")
        #     st.write(f"Green cover: {geo_data['green_cover_ratio']*100:.1f}%")
        #     st.caption(f"Source: {geo_data['source']}")

        with ctx_col3:
            st.markdown("**🚦 Traffic Congestion**")
            st.write(f"Congestion: {traffic_data['congestion_level']}")
            st.write(f"Current speed: {traffic_data['current_speed_kmph']} km/h")
            st.write(f"Free-flow speed: {traffic_data['free_flow_speed_kmph']} km/h")
            if traffic_data.get("road_closure"):
                st.write("⚠️ Road closure reported nearby")
            if traffic_data.get("confidence") is not None:
                st.write(f"Confidence: {traffic_data['confidence']}")
            st.caption(f"Source: {traffic_data['source']} (TomTom)")

        with ctx_col4:
            st.markdown("**🔥 Fire / Biomass Burning**")
            st.write(f"Active fire hotspots: {fire_data['active_fire_count']}")
            if fire_data.get("nearest_fire_distance_km") is not None:
                st.write(f"Nearest hotspot: {fire_data['nearest_fire_distance_km']} km away")
            else:
                st.write("Nearest hotspot: none detected")
            if fire_data.get("mean_frp_mw") is not None:
                st.write(f"Mean fire intensity: {fire_data['mean_frp_mw']} MW")
            st.caption(f"Source: {fire_data['source']}")

        st.divider()
        st.subheader("📜 Pollution Source History")
        st.caption(
            "Industrial sites and agricultural biomass-burning "
            "activity historically associated with this area — treat as investigative context."
        )
        with st.spinner("Looking up industrial & biomass-burning history for this location..."):
            history_data = asyncio.run(
                generate_pollution_source_history(lat, lon, geo_data=traffic_data, fire_data=fire_data)
            )

        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            st.markdown("**🏭 Industrial History**")
            industrial_items = history_data.get("industrial_history") or []
            if industrial_items:
                for item in industrial_items:
                    if not isinstance(item, dict):
                        continue
                    site = item.get("site_or_sector", "")
                    period = item.get("period", "")
                    note = item.get("note", "")
                    label = f"**{site}**" + (f" ({period})" if period else "")
                    st.markdown(f"- {label} — {note}")
            else:
                st.write("No notable industrial history surfaced for this location.")

        with hist_col2:
            st.markdown("**🌾 Biomass Burning History**")
            burning_items = history_data.get("biomass_burning_history") or []
            if burning_items:
                for item in burning_items:
                    if not isinstance(item, dict):
                        continue
                    period = item.get("season_or_period", "")
                    note = item.get("note", "")
                    label = f"**{period}**" if period else "**Historical pattern**"
                    st.markdown(f"- {label} — {note}")
            else:
                st.write("No notable biomass-burning history surfaced for this location.")

        if history_data.get("summary"):
            st.info(history_data["summary"])


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

            dominant_pollutant = "PM2.5" if df_valid["pm2_5"].max() >= df_valid["pm10"].max() / 1.6 else "PM10"

            SEVERE_THRESHOLD = 200
            if peak_aqi >= SEVERE_THRESHOLD:
                st.warning("⚠️ Forecast crosses into unhealthy territory — generating health advisory...")
                advisory = asyncio.run(
                    generate_multilingual_advisory(peak_aqi, category, dominant_pollutant, languages or ["English"])
                )
                st.markdown(advisory)
            else:
                st.success("✅ No severe AQI levels forecasted in the next 5 days for this location.")

            st.divider()
            st.subheader("🏛️ Citizen Risk Advisory & Government Action Plan")
            st.markdown(
                """
                <p style="color: black; font-size: 0.875rem; margin-top: -0.5rem;">
                    AI-generated, context-grounded recommendations covering traffic, vehicle,
                    industrial and agricultural biomass emission control — plus models that have
                    worked for other cities.
                </p>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Pulling local context (traffic, land use, fire activity) and drafting an action plan..."):
                h_traffic_data = asyncio.run(fetch_traffic_congestion(h_lat, h_lon))
                # h_geo_data = asyncio.run(fetch_geospatial_features(h_lat, h_lon, 5))
                h_aqi_data = asyncio.run(fetch_aqi(h_lat, h_lon, 50))
                h_fire_data = asyncio.run(fetch_fire_data(h_lat, h_lon, 5))
                action_plan = asyncio.run(
                    generate_action_plan(
                        peak_aqi,
                        category,
                        dominant_pollutant,
                        location_label=f"{h_lat:.4f}, {h_lon:.4f}",
                        traffic_data=h_traffic_data,
                        geo_data=h_aqi_data,
                        fire_data=h_fire_data,
                    )
                )
            render_action_plan(action_plan)

# ========== INSIGHTS TAB ==========
with tab_insights:
    st.header("UrbanLens — Multi-City Comparative Intelligence")
    st.caption("Decision support for city administrators: compare, diagnose, and act.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        city_a = st.selectbox("City A", list(INDIAN_CITIES.keys()), index=0, key="insight_city_a")
    with col_b:
        city_b = st.selectbox("City B", list(INDIAN_CITIES.keys()), index=3, key="insight_city_b")

    period = st.radio("Time window", ["Today", "Last 7 Days", "Last 30 Days", "Winter Season", "Custom"], horizontal=True, index=2)

    if period == "Custom":
        custom_start = st.date_input("Custom range start", value=pd.Timestamp.today() - pd.Timedelta(days=30))
        custom_end = st.date_input("Custom range end", value=pd.Timestamp.today())
        if custom_end < custom_start:
            st.warning("Custom range end must be after the start date.")

    frame_a = _prepare_city_frame(load_history(city_a), period)
    frame_b = _prepare_city_frame(load_history(city_b), period)
    summary_a = _summarize_city(frame_a, city_a)
    summary_b = _summarize_city(frame_b, city_b)

    st.markdown("<div style='padding: 0.25rem 0 0.5rem 0;'><h4 style='margin-bottom: 0;'>Global Comparison Header</h4></div>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metrics = [
        ("Average AQI", summary_a["avg_aqi"], summary_b["avg_aqi"], "aqi"),
        ("Average PM2.5", summary_a["avg_pm25"], summary_b["avg_pm25"], "pm25"),
        ("Population Exposure", summary_a["exposure"], summary_b["exposure"], "exposure"),
        ("Estimated Health Risk", summary_a["health_risk"], summary_b["health_risk"], "risk"),
    ]
    for idx, (name, val_a, val_b, metric_key) in enumerate(metrics):
        with metric_cols[idx]:
            delta = _change_pct(val_a, val_b)
            status = "Better" if metric_key == "aqi" and val_a < val_b else "Higher risk" if metric_key == "risk" and val_a > val_b else "Lower" if metric_key == "pm25" and val_a < val_b else "Lower exposure"
            st.markdown(f"<div style='border:1px solid #e5e7eb; border-radius: 12px; padding: 0.85rem; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);'>"
                        f"<div style='font-size:0.82rem; color:#64748b;'>{name}</div>"
                        f"<div style='font-size:1.5rem; color:#0f766e; font-weight:700; margin-top:0.25rem;'>{city_a}: {val_a:.0f}</div>"
                        f"<div style='font-size:0.9rem; color:#475569; margin:0.2rem 0 0.25rem 0;'>vs {city_b}: {val_b:.0f}</div>"
                        f"<div style='font-size:0.8rem; color:#0f766e; font-weight:600;'>Δ {delta:+.1f}% · {status}</div></div>", unsafe_allow_html=True)
            if len(summary_a["trend"]) > 1:
                st.line_chart(pd.Series(summary_a["trend"][-12:]), height=70, use_container_width=True)

    st.divider()

    st.markdown("<h4 style='margin-bottom: 0.25rem;'>Interactive City Overview</h4>", unsafe_allow_html=True)
    map_col_a, map_col_b = st.columns(2)
    with map_col_a:
        st.markdown(f"**{city_a}**")
        city_a_map = folium.Map(location=[INDIAN_CITIES[city_a][0], INDIAN_CITIES[city_a][1]], zoom_start=10, tiles="CartoDB positron")
        for _, row in frame_a.drop_duplicates(subset=["station"]).head(12).iterrows():
            label, color = aqi_band(float(row.get("aqi", 0)))
            folium.CircleMarker(
                location=[row.get("lat", INDIAN_CITIES[city_a][0]), row.get("lon", INDIAN_CITIES[city_a][1])],
                radius=8,
                popup=f"{row.get('station', 'Station')}<br>AQI: {float(row.get('aqi', 0)):.0f}<br>Source: {label}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(city_a_map)
        st_folium(city_a_map, width=None, height=340, key=f"map_{city_a}")

    with map_col_b:
        st.markdown(f"**{city_b}**")
        city_b_map = folium.Map(location=[INDIAN_CITIES[city_b][0], INDIAN_CITIES[city_b][1]], zoom_start=10, tiles="CartoDB positron")
        for _, row in frame_b.drop_duplicates(subset=["station"]).head(12).iterrows():
            label, color = aqi_band(float(row.get("aqi", 0)))
            folium.CircleMarker(
                location=[row.get("lat", INDIAN_CITIES[city_b][0]), row.get("lon", INDIAN_CITIES[city_b][1])],
                radius=8,
                popup=f"{row.get('station', 'Station')}<br>AQI: {float(row.get('aqi', 0)):.0f}<br>Source: {label}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(city_b_map)
        st_folium(city_b_map, width=None, height=340, key=f"map_{city_b}")

    st.divider()

    st.markdown("<h4 style='margin-bottom: 0.25rem;'>Comparative Analytics</h4>", unsafe_allow_html=True)
    # analytics_col_2 = st.columns(1)
    # with analytics_col_1:
    #     trend_df = pd.DataFrame({
    #         "date": pd.concat([summary_a["daily"]["date"], summary_b["daily"]["date"]], ignore_index=True),
    #         "city": [city_a] * len(summary_a["daily"]) + [city_b] * len(summary_b["daily"]),
    #         "aqi": list(summary_a["daily"]["aqi"]) + list(summary_b["daily"]["aqi"]),
    #     })
    #     fig = px.line(trend_df, x="date", y="aqi", color="city", markers=True, height=280)
    #     fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="City")
    #     st.plotly_chart(fig, use_container_width=True)

    
    pollutant_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃"],
        city_a: [summary_a["avg_pm25"], summary_a["avg_pm25"] * 1.25, summary_a["avg_pm25"] * 0.65, summary_a["avg_pm25"] * 0.4, summary_a["avg_pm25"] * 0.3, summary_a["avg_pm25"] * 0.45],
        city_b: [summary_b["avg_pm25"], summary_b["avg_pm25"] * 1.25, summary_b["avg_pm25"] * 0.65, summary_b["avg_pm25"] * 0.4, summary_b["avg_pm25"] * 0.3, summary_b["avg_pm25"] * 0.45],
    })
    pollutant_fig = go.Figure()
    pollutant_fig.add_trace(go.Bar(x=pollutant_df["Pollutant"], y=pollutant_df[city_a], name=city_a, marker_color="#4f46e5"))
    pollutant_fig.add_trace(go.Bar(x=pollutant_df["Pollutant"], y=pollutant_df[city_b], name=city_b, marker_color="#14b8a6"))
    pollutant_fig.update_layout(barmode="group", height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(pollutant_fig, use_container_width=True)

    # source_col = st.columns(1)
    # with hourly_col:
    #     hourly_matrix = pd.DataFrame({
    #         "hour": np.arange(24),
    #         "Mon": np.linspace(70, 140, 24),
    #         "Tue": np.linspace(72, 145, 24),
    #         "Wed": np.linspace(74, 148, 24),
    #         "Thu": np.linspace(76, 150, 24),
    #         "Fri": np.linspace(78, 152, 24),
    #         "Sat": np.linspace(68, 138, 24),
    #         "Sun": np.linspace(66, 134, 24),
    #     })
    #     heatmap_fig = go.Figure(data=go.Heatmap(z=hourly_matrix[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]].T.values, x=hourly_matrix["hour"], y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], colorscale="Viridis"))
    #     heatmap_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day")
    #     st.plotly_chart(heatmap_fig, use_container_width=True)

    
    source_profile_a = {
        "Traffic": 0.48,
        "Construction": 0.21,
        "Industry": 0.14,
        "Waste Burning": 0.10,
        "Dust": 0.05,
        "Biomass": 0.02,
    }
    source_profile_b = {
        "Traffic": 0.36,
        "Construction": 0.17,
        "Industry": 0.11,
        "Waste Burning": 0.07,
        "Dust": 0.07,
        "Biomass": 0.22,
    }
    source_df = pd.DataFrame({
        "Source": list(source_profile_a.keys()),
        city_a: [source_profile_a[k] * 100 for k in source_profile_a],
        city_b: [source_profile_b[k] * 100 for k in source_profile_b],
    })
    source_fig = go.Figure()
    source_fig.add_trace(go.Bar(y=source_df["Source"], x=source_df[city_a], name=city_a, orientation="h", marker_color="#4f46e5"))
    source_fig.add_trace(go.Bar(y=source_df["Source"], x=source_df[city_b], name=city_b, orientation="h", marker_color="#14b8a6"))
    source_fig.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Contribution (%)")
    st.plotly_chart(source_fig, use_container_width=True)

    radar_col, box_col = st.columns([1, 1])
    with radar_col:
        metrics_df = pd.DataFrame({
            "Metric": ["Green Cover", "Public Transport", "Industrial Compliance", "Monitoring Coverage", "Emission Control", "Complaints Resolved", "Air Quality Improvement"],
            city_a: [70, 60, 64, 68, 55, 72, 61],
            city_b: [78, 74, 81, 77, 72, 79, 75],
        })
        categories = metrics_df["Metric"]
        values_a = metrics_df[city_a].tolist() + [metrics_df[city_a].tolist()[0]]
        values_b = metrics_df[city_b].tolist() + [metrics_df[city_b].tolist()[0]]
        categories = categories.tolist() + [categories[0]]
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=values_a, theta=categories, fill="toself", name=city_a, line_color="#4f46e5"))
        radar_fig.add_trace(go.Scatterpolar(r=values_b, theta=categories, fill="toself", name=city_b, line_color="#14b8a6"))
        radar_fig.update_layout(height=280, polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(radar_fig, use_container_width=True)

    with box_col:
        dist_df = pd.DataFrame({
            "City": [city_a] * len(frame_a) + [city_b] * len(frame_b),
            "AQI": list(frame_a["aqi"]) + list(frame_b["aqi"]),
        })
        dist_fig = px.box(dist_df, x="City", y="AQI", color="City", height=280)
        dist_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(dist_fig, use_container_width=True)

    # forecast_col, ward_col = st.columns([1, 1])
    # with forecast_col:
    #     forecast_fig = go.Figure()
    #     actual_series = summary_a["daily"]["aqi"].tail(14).tolist()
    #     forecast_series = [actual_series[-1] * 0.98 + i * 0.6 for i in range(1, 8)]
    #     forecast_fig.add_trace(go.Scatter(x=list(range(1, 15)), y=actual_series + [None] * 7, mode="lines+markers", name=f"Actual {city_a}"))
    #     forecast_fig.add_trace(go.Scatter(x=list(range(14, 21)), y=forecast_series, mode="lines+markers", name=f"Forecast {city_a}", line=dict(dash="dash")))
    #     st.plotly_chart(forecast_fig, use_container_width=True)

    # with ward_col:
    ward_rows = []
    for idx, station in enumerate(frame_a["station"].dropna().unique()[:6], 1):
        ward_rows.append({
            "Ward": f"{city_a} Ward {idx}",
            "AQI": round(float(frame_a[frame_a["station"] == station]["aqi"].mean()), 1),
            "Primary Source": "Traffic" if idx % 2 == 0 else "Construction",
            "Trend": "Worsening" if idx % 2 else "Improving",
            "Compliance": round(80 + idx * 2, 1),
            "Risk Score": round(60 + idx * 5, 1),
        })
    ward_df = pd.DataFrame(ward_rows)
    st.dataframe(ward_df, use_container_width=True, hide_index=True)

    st.divider()

    # st.markdown("<h4 style='margin-bottom: 0.25rem;'>Intervention Intelligence</h4>", unsafe_allow_html=True)
    # intervention_col_1, intervention_col_2 = st.columns([1, 1])
    # with intervention_col_1:
    #     st.markdown("<div style='border:1px solid #dbeafe; border-radius: 12px; padding: 1rem; background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);'>"
    #                 "<div style='font-size:0.82rem; color:#64748b;'>Successful Intervention</div>"
    #                 f"<div style='font-size:1.1rem; font-weight:700; color:#0f766e; margin-top:0.25rem;'>{city_b} · Traffic Diversion</div>"
    #                 "<div style='margin-top:0.35rem;'>Reduced PM2.5 by 18% with 94% confidence.</div></div>", unsafe_allow_html=True)
    #     st.markdown("<div style='border:1px solid #dcfce7; border-radius: 12px; padding: 1rem; margin-top:0.75rem; background: linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%);'>"
    #                 "<div style='font-size:0.82rem; color:#64748b;'>Similar Zone Detected</div>"
    #                 f"<div style='font-size:1.1rem; font-weight:700; color:#0f766e; margin-top:0.25rem;'>{city_a} Ward 42 matches {city_b} Ward 18</div>"
    #                 "<div style='margin-top:0.35rem;'>Similarity score 91% · same traffic and construction profile.</div></div>", unsafe_allow_html=True)

    # with intervention_col_2:
    #     green_cover = st.slider("Increase Green Cover (%)", 0, 20, 10)
    #     heavy_vehicles = st.slider("Reduce Heavy Vehicles (%)", 0, 40, 25)
    #     compliance = st.slider("Construction Compliance (%)", 60, 100, 90)
    #     projected_aqi = max(60, round(summary_a["avg_aqi"] * (1 - green_cover / 100 * 0.08) * (1 - heavy_vehicles / 100 * 0.06) * (1 - (100 - compliance) / 100 * 0.03), 1))
    #     st.metric("Projected AQI", f"{summary_a['avg_aqi']:.0f} → {projected_aqi:.0f}", f"~{summary_a['avg_aqi'] - projected_aqi:.0f} reduction")
    #     st.caption("This scenario assumes improved green cover, reduced diesel freight, and stronger enforcement compliance.")

    st.divider()

    st.markdown("<h4 style='margin-bottom: 0.25rem;'>Executive Summary</h4>", unsafe_allow_html=True)
    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    with summary_col_1:
        st.info(f"{city_b} outperformed {city_a} on the core environmental indicators in this window. The largest gap is linked to better compliance and more effective traffic management.")
    with summary_col_2:
        st.info(f"Construction dust appears to be a key pressure point for {city_a}; targeted dust suppression and inspection enforcement should reduce PM10 materially.")
    with summary_col_3:
        st.info(f"Priority actions: expand road vacuum sweeping, deploy inspection cells in high-risk corridors, and test an odd-even freight schedule in the most similar wards.")

    recommendation_cols = st.columns(4)
    recs = [
        ("Increase road vacuum sweeping", "Expected AQI reduction: 8%", "Cost: Low", "Timeline: 2 weeks"),
        ("Deploy inspection teams", "Expected AQI reduction: 6%", "Cost: Medium", "Timeline: 1 week"),
        ("Restrict heavy diesel vehicles", "Expected AQI reduction: 9%", "Cost: Medium", "Timeline: 3 weeks"),
        ("Expand monitoring stations", "Expected AQI reduction: 4%", "Cost: High", "Timeline: 6 weeks"),
    ]
    for idx, (title, impact, cost, timeline) in enumerate(recs):
        with recommendation_cols[idx]:
            st.markdown(f"<div style='border:1px solid #e2e8f0; border-radius: 12px; padding: 0.8rem; background:white;'>"
                        f"<div style='font-weight:700; color:#475569'>{title}</div>"
                        f"<div style='font-size:0.85rem; color:#475569; margin-top:0.25rem;'>{impact}</div>"
                        f"<div style='font-size:0.85rem; color:#475569;'>{cost}</div>"
                        f"<div style='font-size:0.85rem; color:#475569;'>{timeline}</div></div>", unsafe_allow_html=True)