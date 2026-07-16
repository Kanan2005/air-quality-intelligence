"""
Deterministic synthetic data generators.

These generators are used as a graceful fallback whenever a live external
API is unreachable (no internet, rate-limited, missing API key, etc.) so
that the platform remains fully functional and demo-able offline.

Determinism is achieved by seeding NumPy's RNG from a hash of the
(latitude, longitude) pair, meaning the same location will always yield the
same synthetic data within a single process -- important for believable,
reproducible demo behaviour.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict

import numpy as np


def _seed_from_coords(lat: float, lon: float, salt: str = "") -> int:
    key = f"{round(lat, 4)}:{round(lon, 4)}:{salt}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def _rng(lat: float, lon: float, salt: str = "") -> np.random.Generator:
    return np.random.default_rng(_seed_from_coords(lat, lon, salt))


def mock_aqi(lat: float, lon: float) -> Dict:
    """Generate realistic synthetic AQI / pollutant concentration data."""
    rng = _rng(lat, lon, "aqi")

    # Urban cores (roughly dense city centers) tend to have higher baseline
    # pollution in this synthetic model -- we approximate "urban-ness" using
    # a smooth pseudo-random field derived from the coordinates themselves.
    urban_factor = 0.5 + 0.5 * math.sin(lat * 12.9) * math.cos(lon * 7.3)
    base_pm25 = 60 + 140 * abs(urban_factor) + rng.normal(0, 15)
    base_pm25 = float(np.clip(base_pm25, 8, 420))

    pm10 = base_pm25 * rng.uniform(1.4, 1.9)
    no2 = float(np.clip(rng.normal(35, 12) + urban_factor * 20, 2, 180))
    so2 = float(np.clip(rng.normal(12, 6), 1, 80))
    co = float(np.clip(rng.normal(0.8, 0.3), 0.1, 5))
    o3 = float(np.clip(rng.normal(40, 15), 5, 150))

    aqi = _pm25_to_aqi(base_pm25)

    return {
        "aqi": aqi,
        "pm25": round(base_pm25, 1),
        "pm10": round(pm10, 1),
        "no2": round(no2, 1),
        "so2": round(so2, 1),
        "co": round(co, 2),
        "o3": round(o3, 1),
        "category": _aqi_category(aqi),
        "source": "synthetic_mock",
    }


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _pm25_to_aqi(pm25: float) -> int | None:
    """Convert PM2.5 concentration (ug/m3) to Indian-style AQI (simplified breakpoints).

    Returns None if pm25 is missing/NaN or out of the supported physical range,
    rather than silently defaulting to a "Severe" (500) reading -- a missing
    value is not evidence of severe pollution.
    """
    if _is_missing(pm25):
        return None
    breakpoints = [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = (a_hi - a_lo) / (c_hi - c_lo) * (pm25 - c_lo) + a_lo
            return int(round(aqi))
    return 500 if pm25 > 500 else None


def _pm10_to_aqi(pm10: float) -> int | None:
    """Convert PM10 concentration (ug/m3) to Indian-style AQI (CPCB breakpoints).

    Used as a fallback sub-index when PM2.5 is unavailable for an hour but
    PM10 is present -- CPCB's real India AQI is the max sub-index across
    pollutants, so PM10 is a legitimate independent signal, not a proxy.
    """
    if _is_missing(pm10):
        return None
    breakpoints = [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 500, 400, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= pm10 <= c_hi:
            aqi = (a_hi - a_lo) / (c_hi - c_lo) * (pm10 - c_lo) + a_lo
            return int(round(aqi))
    return 500 if pm10 > 500 else None


def india_aqi_from_pollutants(pm25: float, pm10: float) -> int | None:
    """Combine PM2.5 and PM10 sub-indices into a single India AQI reading.

    CPCB's real India AQI is defined as the max sub-index across measured
    pollutants. Here we only have PM2.5 and PM10 available from the
    hyperlocal forecast source, so we take the max of whichever sub-index(es)
    are computable for this hour. Returns None only if neither pollutant has
    a usable reading.
    """
    candidates = [aqi for aqi in (_pm25_to_aqi(pm25), _pm10_to_aqi(pm10)) if aqi is not None]
    return max(candidates) if candidates else None


def _aqi_category(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def mock_weather(lat: float, lon: float) -> Dict:
    """Generate synthetic weather data influencing pollutant dispersion."""
    rng = _rng(lat, lon, "weather")

    season_temp = 22 + 10 * math.sin(lat * 3.1)
    temperature = float(np.clip(season_temp + rng.normal(0, 4), -10, 48))
    humidity = float(np.clip(rng.normal(55, 20), 10, 98))
    wind_speed = float(np.clip(rng.gamma(2, 1.2), 0.2, 12))
    wind_direction = float(rng.uniform(0, 360))
    pressure = float(np.clip(rng.normal(1013, 6), 985, 1035))

    return {
        "temperature_c": round(temperature, 1),
        "humidity_pct": round(humidity, 1),
        "wind_speed_mps": round(wind_speed, 2),
        "wind_direction_deg": round(wind_direction, 1),
        "pressure_hpa": round(pressure, 1),
        "source": "synthetic_mock",
    }


def mock_geospatial(lat: float, lon: float, radius_km: float) -> Dict:
    """Generate synthetic land-use / road-network context."""
    rng = _rng(lat, lon, "geo")

    urban_factor = abs(0.5 + 0.5 * math.sin(lat * 12.9) * math.cos(lon * 7.3))

    road_density = float(np.clip(rng.normal(4 + urban_factor * 10, 2), 0.1, 25))
    major_roads = int(np.clip(rng.poisson(2 + urban_factor * 6), 0, 40))
    industrial_ratio = float(np.clip(rng.beta(2, 6) * urban_factor * 1.5, 0, 0.8))
    construction_sites = float(np.clip(rng.poisson(1 + urban_factor * 4), 0, 25))
    green_cover = float(np.clip(rng.beta(2, 3) * (1.2 - urban_factor), 0.02, 0.9))
    built_up = float(np.clip(urban_factor * rng.uniform(0.5, 1.0), 0.05, 0.95))

    return {
        "road_density_km_per_km2": round(road_density, 2),
        "major_road_count": major_roads,
        "industrial_area_ratio": round(industrial_ratio, 3),
        "construction_site_count": round(construction_sites, 1),
        "green_cover_ratio": round(green_cover, 3),
        "built_up_ratio": round(built_up, 3),
        "source": "synthetic_mock",
    }


def mock_fire_data(lat: float, lon: float, radius_km: float) -> Dict:
    """Generate synthetic NASA FIRMS-style active-fire / biomass-burning data."""
    rng = _rng(lat, lon, "fire")

    # Biomass burning is seasonal/regional in reality; here we approximate
    # with a low base rate that occasionally spikes.
    spike = rng.random() < 0.18
    fire_count = int(rng.poisson(4)) if spike else int(rng.poisson(0.3))
    nearest_distance = None
    mean_frp = None
    if fire_count > 0:
        nearest_distance = round(float(rng.uniform(0.2, radius_km)), 2)
        mean_frp = round(float(rng.gamma(2, 8)), 1)

    return {
        "active_fire_count": fire_count,
        "nearest_fire_distance_km": nearest_distance,
        "mean_frp_mw": mean_frp,
        "source": "synthetic_mock",
    }


SAMPLE_LOCATIONS = [
    {"name": "Anand Vihar, Delhi", "latitude": 28.6469, "longitude": 77.3152},
    {"name": "Connaught Place, Delhi", "latitude": 28.6315, "longitude": 77.2167},
    {"name": "Bandra, Mumbai", "latitude": 19.0596, "longitude": 72.8295},
    {"name": "Whitefield, Bengaluru", "latitude": 12.9698, "longitude": 77.7500},
    {"name": "Karve Road, Pune", "latitude": 18.5040, "longitude": 73.8154},
    {"name": "Sector 62, Noida", "latitude": 28.6274, "longitude": 77.3716},
    {"name": "Howrah, Kolkata", "latitude": 22.5958, "longitude": 88.2636},
]