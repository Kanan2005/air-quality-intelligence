"""
Pollution source attribution — HEURISTIC / RULE-BASED PROXY.

Honest note for your deck: real source attribution (the problem statement's
"Geospatial Pollution Source Attribution Engine") needs land-use maps,
traffic sensor feeds, and industrial stack registries — none of which have
a free, national, machine-readable API in India. Faking a black-box "ML
model" here would not survive judge questioning.

Instead, this module attributes likely source category using a transparent,
explainable rule set driven purely by TIME PATTERNS in the AQI signal
itself (which we do have, for free, in real-time):

  - Sharp double peak at ~8-10am & ~7-10pm, weekday > weekend -> Traffic
  - Elevated and flat through 10am-6pm on weekdays -> Construction/Industrial
  - Persistently high with no clear diurnal shape -> Regional/Background
    (crop burning, dust, meteorology-driven)

This is defensible as a v1 "attribution confidence score" and is exactly
the kind of thing you flag to judges as: "this is our free-data MVP;
production version would ingest OSM land-use polygons + VIIRS fire/thermal
anomaly data (also free, via NASA FIRMS) to firm up confidence."

NASA FIRMS (fire/thermal anomalies) IS actually free and real:
https://firms.modaps.eosdis.nasa.gov/api/ — worth wiring in as a real
stretch-goal improvement (see docs/architecture.md).
"""

import pandas as pd
import numpy as np


def attribute_station(hourly_df: pd.DataFrame) -> dict:
    """
    hourly_df: single station's data with columns [timestamp, aqi],
    ideally >= 7 days of hourly readings.
    Returns a dict of source_category -> confidence (0-1), plus rationale.
    """
    df = hourly_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    hourly_mean = df.groupby("hour")["aqi"].mean()
    morning_peak = hourly_mean.loc[7:10].mean()
    evening_peak = hourly_mean.loc[19:22].mean()
    midday_flat = hourly_mean.loc[11:17].mean()
    overnight = hourly_mean.loc[[0, 1, 2, 3, 4]].mean()

    weekday_mean = df[~df["is_weekend"]]["aqi"].mean()
    weekend_mean = df[df["is_weekend"]]["aqi"].mean()
    weekday_ratio = weekday_mean / weekend_mean if weekend_mean else 1.0

    peak_prominence = ((morning_peak + evening_peak) / 2) - overnight
    diurnal_range = hourly_mean.max() - hourly_mean.min()

    scores = {"traffic": 0.0, "construction_industrial": 0.0, "regional_background": 0.0}

    # Traffic signature: strong double peak + weekday-heavy
    if peak_prominence > 0.15 * hourly_mean.mean() and weekday_ratio > 1.05:
        scores["traffic"] = min(1.0, 0.5 + peak_prominence / hourly_mean.mean())

    # Construction/industrial: sustained midday elevation, less time-of-day swing
    if midday_flat > 0.85 * hourly_mean.max() and diurnal_range < 0.4 * hourly_mean.mean():
        scores["construction_industrial"] = min(1.0, midday_flat / hourly_mean.mean())

    # Regional/background: high baseline, low diurnal range, weekday≈weekend
    if diurnal_range < 0.25 * hourly_mean.mean() and abs(weekday_ratio - 1.0) < 0.05:
        scores["regional_background"] = min(1.0, 0.6 + (0.25 - diurnal_range / hourly_mean.mean()))

    total = sum(scores.values()) or 1.0
    normalized = {k: round(v / total, 2) for k, v in scores.items()}

    top_source = max(normalized, key=normalized.get) if any(normalized.values()) else "inconclusive"

    return {
        "attribution": normalized,
        "top_source": top_source,
        "rationale": {
            "morning_peak_aqi": round(morning_peak, 1),
            "evening_peak_aqi": round(evening_peak, 1),
            "overnight_baseline_aqi": round(overnight, 1),
            "weekday_vs_weekend_ratio": round(weekday_ratio, 2),
        },
        "note": "Rule-based proxy on time-pattern only — see module docstring for production upgrade path.",
    }


if __name__ == "__main__":
    df = pd.read_csv("../data/sample_aqi_data.csv", parse_dates=["timestamp"])
    station = df[df["station"] == df["station"].iloc[0]]
    result = attribute_station(station)
    import json
    print(json.dumps(result, indent=2))
