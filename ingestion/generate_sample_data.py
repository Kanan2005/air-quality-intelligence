"""
Generates a realistic synthetic hourly PM2.5/AQI dataset for multiple Indian
cities and stations, spanning 30 days. This exists so that:

  1. The dashboard and forecasting model always have something to demo on,
     even with zero internet access (judges' wifi, offline demo, etc.)
  2. You have a drop-in-compatible schema to validate your pipeline against
     before wiring up the live OpenAQ key.

Replace this with real ingestion/openaq_client.py output once you have an
API key. Schema is identical either way.

Diurnal + weekly patterns are modeled roughly on published CPCB behaviour:
morning/evening traffic peaks, winter-like baseline pollution, weekday >
weekend traffic contribution.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from openaq_client import pm25_to_aqi, INDIAN_CITIES

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv"

# Baseline PM2.5 severity per city (reflects rough real-world ordering from
# CPCB 2024-25 reporting: Delhi worst, Bengaluru/Chennai comparatively better)
CITY_BASELINE = {
    "Delhi": 140,
    "Mumbai": 75,
    "Kolkata": 95,
    "Bengaluru": 55,
    "Chennai": 60,
    "Hyderabad": 70,
    "Pune": 65,
    "Agra": 110,
}

STATIONS_PER_CITY = 4
DAYS_OF_HISTORY = 30
RNG = np.random.default_rng(42)


def diurnal_multiplier(hour: int) -> float:
    """Traffic-driven double peak: ~8-10am and ~7-10pm, low overnight/midday."""
    morning_peak = np.exp(-((hour - 9) ** 2) / (2 * 2.5 ** 2))
    evening_peak = np.exp(-((hour - 20) ** 2) / (2 * 3 ** 2))
    return 0.65 + 0.55 * (morning_peak + evening_peak)


def weekday_multiplier(dow: int) -> float:
    # dow: Monday=0 ... Sunday=6. Slightly lower on Sundays (less traffic/construction).
    return 0.85 if dow == 6 else 1.0


def generate():
    rows = []
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=DAYS_OF_HISTORY)
    timestamps = pd.date_range(start, end, freq="h")

    for city, (lat, lon) in INDIAN_CITIES.items():
        baseline = CITY_BASELINE[city]
        for s in range(STATIONS_PER_CITY):
            station_name = f"{city}-Station-{s+1}"
            # jitter station location slightly around city centroid (~5-10km)
            s_lat = lat + RNG.uniform(-0.06, 0.06)
            s_lon = lon + RNG.uniform(-0.06, 0.06)
            station_bias = RNG.uniform(0.85, 1.2)  # some stations run hotter (near industry/traffic)

            for ts in timestamps:
                dm = diurnal_multiplier(ts.hour)
                wm = weekday_multiplier(ts.weekday())
                noise = RNG.normal(1.0, 0.12)
                pm25 = max(5.0, baseline * dm * wm * station_bias * noise)
                rows.append({
                    "city": city,
                    "station": station_name,
                    "lat": round(s_lat, 5),
                    "lon": round(s_lon, 5),
                    "parameter": "pm25",
                    "value": round(pm25, 1),
                    "unit": "µg/m³",
                    "timestamp": ts,
                })

    df = pd.DataFrame(rows)
    df["aqi"] = df["value"].apply(pm25_to_aqi)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUT_PATH}")
    print(df.groupby("city")["aqi"].mean().round(1))


if __name__ == "__main__":
    generate()
