"""
OpenAQ v3 ingestion client.

OpenAQ (https://openaq.org) aggregates CPCB and other government monitoring
station data for free. As of v3, a free API key is required.

Get a free key in 30 seconds at: https://explore.openaq.org/register
Then set it as an environment variable:
    export OPENAQ_API_KEY="your_key_here"

This module has NO paid dependency. If no key / no internet is available
(e.g. running inside a sandboxed demo), it transparently falls back to the
bundled synthetic-but-realistic sample dataset in data/sample_aqi_data.csv
so the rest of the pipeline (forecasting, dashboard) always has something
to run against.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://api.openaq.org/v3"
SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv"

# A handful of major Indian cities with approximate lat/lon centroids,
# used to query OpenAQ's "locations near a point" endpoint.
INDIAN_CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Agra": (27.1767, 78.0081),
}

# CPCB / NAQI breakpoints (PM2.5, µg/m3) -> AQI sub-index, simplified linear
# interpolation per CPCB National Air Quality Index methodology.
PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, 380, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration (µg/m3) to CPCB National AQI sub-index."""
    if pm25 is None or pd.isna(pm25):
        return None
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round(i_lo + (i_hi - i_lo) * (pm25 - c_lo) / (c_hi - c_lo), 1)
    if pm25 > 380:
        return 500.0
    return 0.0


class OpenAQClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

    def _get(self, path: str, params: dict, retries: int = 3):
        url = f"{BASE_URL}{path}"
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:  # rate limited, back off
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                if attempt == retries - 1:
                    return None
                time.sleep(1)
        return None

    def get_locations_near(self, lat: float, lon: float, radius_m: int = 15000, limit: int = 25):
        """Find monitoring stations near a point."""
        data = self._get(
            "/locations",
            {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": limit},
        )
        if not data:
            return []
        return data.get("results", [])

    def get_latest_measurements(self, location_id: int):
        data = self._get(f"/locations/{location_id}/latest", {})
        if not data:
            return []
        return data.get("results", [])

    def get_sensor_history(self, sensor_id: int, date_from: str, date_to: str, limit: int = 1000):
        """Hourly historical measurements for a specific sensor (parameter+location)."""
        data = self._get(
            f"/sensors/{sensor_id}/measurements",
            {"date_from": date_from, "date_to": date_to, "limit": limit},
        )
        if not data:
            return []
        return data.get("results", [])

    def fetch_city_snapshot(self, city: str) -> pd.DataFrame:
        """
        High-level convenience method: pulls current readings for all
        stations found near a named city. Returns a tidy DataFrame with
        columns: city, station, lat, lon, parameter, value, unit, aqi, timestamp.
        Falls back to sample data if the API is unreachable / no key set.
        """
        if city not in INDIAN_CITIES:
            raise ValueError(f"Unknown city '{city}'. Choose from {list(INDIAN_CITIES)}")

        rows = []
        if self.api_key:
            lat, lon = INDIAN_CITIES[city]
            locations = self.get_locations_near(lat, lon)
            for loc in locations:
                loc_id = loc.get("id")
                station_name = loc.get("name", f"station_{loc_id}")
                coords = loc.get("coordinates", {})
                latest = self.get_latest_measurements(loc_id)
                for m in latest:
                    param = m.get("parameter", {}).get("name")
                    value = m.get("value")
                    unit = m.get("parameter", {}).get("units")
                    ts = m.get("datetime", {}).get("utc")
                    rows.append({
                        "city": city,
                        "station": station_name,
                        "lat": coords.get("latitude"),
                        "lon": coords.get("longitude"),
                        "parameter": param,
                        "value": value,
                        "unit": unit,
                        "timestamp": ts,
                    })

        if not rows:
            # Fallback: bundled sample data (still realistic, city-filtered)
            return self._load_sample(city)

        df = pd.DataFrame(rows)
        pm25 = df[df["parameter"].str.lower() == "pm25"].copy()
        pm25["aqi"] = pm25["value"].apply(pm25_to_aqi)
        return pm25

    @staticmethod
    def _load_sample(city: str) -> pd.DataFrame:
        if not SAMPLE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"No API key set and no sample data found at {SAMPLE_DATA_PATH}. "
                "Run ingestion/generate_sample_data.py first."
            )
        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["timestamp"])
        return df[df["city"] == city].copy()


if __name__ == "__main__":
    client = OpenAQClient()
    for city in ["Delhi", "Mumbai"]:
        snap = client.fetch_city_snapshot(city)
        print(f"\n{city}: {len(snap)} readings")
        print(snap.head())
