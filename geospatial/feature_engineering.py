"""Feature engineering: transforms raw multi-source data into a model-ready feature vector."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

FEATURE_NAMES: List[str] = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "road_density_km_per_km2",
    "major_road_count",
    "industrial_area_ratio",
    "construction_site_count",
    "green_cover_ratio",
    "built_up_ratio",
    "active_fire_count",
    "mean_frp_mw",
    "wind_speed_mps",
    "humidity_pct",
    "temperature_c",
]


def build_feature_vector(
    aqi_data: Dict, weather_data: Dict, geo_data: Dict, fire_data: Dict
) -> np.ndarray:
    """Assemble a single feature row (order must match FEATURE_NAMES)."""
    row = [
        aqi_data.get("pm25", 0) or 0,
        aqi_data.get("pm10", 0) or 0,
        aqi_data.get("no2", 0) or 0,
        aqi_data.get("so2", 0) or 0,
        aqi_data.get("co", 0) or 0,
        aqi_data.get("o3", 0) or 0,
        geo_data.get("road_density_km_per_km2", 0) or 0,
        geo_data.get("major_road_count", 0) or 0,
        geo_data.get("industrial_area_ratio", 0) or 0,
        geo_data.get("construction_site_count", 0) or 0,
        geo_data.get("green_cover_ratio", 0) or 0,
        geo_data.get("built_up_ratio", 0) or 0,
        fire_data.get("active_fire_count", 0) or 0,
        fire_data.get("mean_frp_mw", 0) or 0,
        weather_data.get("wind_speed_mps", 0) or 0,
        weather_data.get("humidity_pct", 0) or 0,
        weather_data.get("temperature_c", 0) or 0,
    ]
    return np.array(row, dtype=float).reshape(1, -1)


def build_feature_dict(
    aqi_data: Dict, weather_data: Dict, geo_data: Dict, fire_data: Dict
) -> Dict[str, float]:
    """Same as build_feature_vector but as a labelled dict (useful for logging/explainability)."""
    vec = build_feature_vector(aqi_data, weather_data, geo_data, fire_data).flatten()
    return dict(zip(FEATURE_NAMES, vec.tolist()))


def compute_data_quality_flags(
    aqi_data: Dict, weather_data: Dict, geo_data: Dict, fire_data: Dict
) -> Dict[str, bool]:
    """Flag which data sources were live vs. synthetic -- feeds into confidence scoring."""
    return {
        "aqi_live": aqi_data.get("source") not in (None, "synthetic_mock"),
        "weather_live": weather_data.get("source") not in (None, "synthetic_mock"),
        "geospatial_live": geo_data.get("source") not in (None, "synthetic_mock"),
        "fire_live": fire_data.get("source") not in (None, "synthetic_mock"),
    }
