"""
Government AQI data ingestion client.

This project now uses the free government air-quality API provided by the
Indian government data portal instead of the earlier OpenAQ-based workflow.

The endpoint is:
    https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69

Set a custom API key with:
    export GOVT_AQI_API_KEY="your_key_here"

If the API is unavailable or returns no records, the module falls back to the
bundled synthetic sample dataset so the rest of the pipeline still runs.
"""

import os
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import pandas as pd
from pathlib import Path

BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
DEFAULT_API_KEY = "579b464db66ec23bdd0000013b08f5e6a3a94faf57869ecbf2f79c80"
SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv"
LOCAL_XML_PATHS = [
    Path("/Users/kananagarwal/Downloads/datasets_of_real_time_air_quality_index_804641/data/data_aqi_cpcb.xml"),
    Path(__file__).resolve().parent.parent / "data" / "data_aqi_cpcb.xml",
]

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

CITY_TO_STATE = {
    "Delhi": "Delhi",
    "Mumbai": "Maharashtra",
    "Kolkata": "West Bengal",
    "Bengaluru": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Hyderabad": "Telangana",
    "Pune": "Maharashtra",
    "Agra": "Uttar Pradesh",
}

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


class GovernmentAQIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOVT_AQI_API_KEY") or DEFAULT_API_KEY

    def _get(self, params: dict, retries: int = 3):
        request_params = {"api-key": self.api_key, "format": "json", **params}
        url = f"{BASE_URL}?{urlencode(request_params)}"
        for attempt in range(retries):
            try:
                req = Request(url, headers={"User-Agent": "air-quality-intelligence/1.0"})
                with urlopen(req, timeout=20) as resp:
                    payload = resp.read().decode("utf-8")
                    return json.loads(payload)
            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(1)
        return None

    def get_latest_measurements(self, city: str):
        state = CITY_TO_STATE.get(city, city)
        data = self._get({"limit": 1000, "filters[state]": state})
        if not data:
            return []
        return data.get("records", [])

    def fetch_city_snapshot(self, city: str) -> pd.DataFrame:
        if city not in INDIAN_CITIES:
            raise ValueError(f"Unknown city '{city}'. Choose from {list(INDIAN_CITIES)}")

        rows = []
        if self.api_key:
            records = self.get_latest_measurements(city)
            for record in records:
                row = self._normalize_record(city, record)
                if row is not None:
                    rows.append(row)

        if not rows:
            rows = self._load_local_xml_snapshot(city)

        if not rows:
            return self._load_sample(city)

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["aqi"]).copy()
        if df.empty:
            return df

        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["aqi"]).copy()
        if df.empty:
            return df

        df = df.sort_values(["station", "aqi"], ascending=[True, False])
        df = df.drop_duplicates(subset=["station"], keep="first")
        return df

    def _normalize_record(self, city: str, record: dict) -> dict:
        station = record.get("station") or record.get("station_name") or record.get("name") or f"{city}-station"
        latitude = record.get("latitude") or record.get("lat")
        longitude = record.get("longitude") or record.get("lon")
        timestamp = record.get("last_update") or record.get("last_updated") or record.get("date") or record.get("timestamp")
        parameter = record.get("pollutant_id") or record.get("parameter") or record.get("pollutant") or "AQI"

        raw_value = None
        for key in ["avg_value", "pollutant_avg", "aqi", "aqi_value", "pollutant_value", "value"]:
            if key in record and record.get(key) not in (None, ""):
                raw_value = record.get(key)
                break

        if raw_value is None:
            return None

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        if str(parameter).upper() in {"PM2.5", "PM25"}:
            aqi = pm25_to_aqi(value)
        else:
            aqi = value

        return {
            "city": record.get("city") or city,
            "station": station,
            "lat": latitude,
            "lon": longitude,
            "parameter": str(parameter),
            "value": value,
            "unit": record.get("unit") or ("AQI" if parameter in {"AQI", "aqi"} else "µg/m³"),
            "aqi": aqi,
            "timestamp": timestamp,
        }

    @staticmethod
    def _load_local_xml_snapshot(city: str) -> list:
        for path in LOCAL_XML_PATHS:
            if not path.exists():
                continue
            try:
                root = ET.parse(path).getroot()
                rows = []
                for state in root.findall(".//State"):
                    for city_node in state.findall(".//City"):
                        city_name = city_node.attrib.get("id")
                        if city_name and city_name.lower() != city.lower().lower():
                            continue
                        for station in city_node.findall(".//Station"):
                            station_id = station.attrib.get("id")
                            timestamp = station.attrib.get("lastupdate")
                            lat = station.attrib.get("latitude")
                            lon = station.attrib.get("longitude")
                            aqi_node = station.find("Air_Quality_Index")
                            if aqi_node is None:
                                continue
                            try:
                                aqi_value = float(aqi_node.attrib.get("Value", "nan"))
                            except (TypeError, ValueError):
                                continue
                            if pd.isna(aqi_value):
                                continue
                            rows.append({
                                "city": city_name or city,
                                "station": station_id,
                                "lat": lat,
                                "lon": lon,
                                "parameter": "AQI",
                                "value": aqi_value,
                                "unit": "AQI",
                                "aqi": aqi_value,
                                "timestamp": datetime.strptime(timestamp, "%d-%m-%Y %H:%M:%S") if timestamp else None,
                            })
                if rows:
                    return rows
            except Exception:
                continue
        return []

    @staticmethod
    def _load_sample(city: str) -> pd.DataFrame:
        if not SAMPLE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"No API key set and no sample data found at {SAMPLE_DATA_PATH}. "
                "Run ingestion/generate_sample_data.py first."
            )
        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["timestamp"])
        return df[df["city"] == city].copy()


OpenAQClient = GovernmentAQIClient


if __name__ == "__main__":
    client = GovernmentAQIClient()
    for city in ["Delhi", "Mumbai"]:
        snap = client.fetch_city_snapshot(city)
        print(f"\n{city}: {len(snap)} readings")
        print(snap.head())
