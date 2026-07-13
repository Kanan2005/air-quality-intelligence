"""Confidence scoring and rule-based natural-language explanation generation."""
from __future__ import annotations

from typing import Dict, List, Tuple

SOURCE_DISPLAY = {
    "traffic": "heavy traffic congestion",
    "construction": "nearby construction activity",
    "industries": "nearby industrial emissions",
    "biomass_burning": "biomass / crop-residue burning",
}


def compute_confidence_score(
    tree_agreement_score: float,
    data_quality_flags: Dict[str, bool],
    aqi_data: Dict,
) -> float:
    """
    Combine multiple signals into a single 0-100 confidence score:
      - Model (ensemble tree agreement): 50% weight
      - Live vs. synthetic data source coverage: 35% weight
      - AQI reading plausibility / extremity: 15% weight
    """
    live_fraction = sum(1 for v in data_quality_flags.values() if v) / max(
        1, len(data_quality_flags)
    )

    pm25 = aqi_data.get("pm25", 0) or 0
    # Extremely low or extremely high PM2.5 values are less common / less reliable
    plausibility = 1.0
    if pm25 < 3 or pm25 > 500:
        plausibility = 0.5

    score = (
        0.50 * tree_agreement_score
        + 0.35 * live_fraction
        + 0.15 * plausibility
    ) * 100.0

    return round(float(max(35.0, min(99.0, score))), 1)


def build_explanation(
    source_contribution: Dict[str, float],
    aqi_data: Dict,
    weather_data: Dict,
    geo_data: Dict,
    fire_data: Dict,
) -> Tuple[str, str, List[str]]:
    """
    Generate a human-readable explanation, the dominant source label, and a
    short list of actionable recommendations.
    """
    ranked = sorted(source_contribution.items(), key=lambda kv: -kv[1])
    dominant_source, dominant_pct = ranked[0]
    top_two = ranked[:2]

    category = aqi_data.get("category", "Moderate")
    pm25 = aqi_data.get("pm25", 0)

    reasons = []
    for label, pct in top_two:
        if pct >= 15:
            reasons.append(SOURCE_DISPLAY.get(label, label))

    if not reasons:
        reasons = [SOURCE_DISPLAY.get(dominant_source, dominant_source)]

    reason_text = " and ".join(reasons) if len(reasons) <= 2 else ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    explanation = (
        f"{category} air quality (PM2.5 ~{pm25:.0f} \u00b5g/m\u00b3) is primarily attributed to "
        f"{reason_text}. This assessment combines road-network density, industrial and "
        f"construction land-use context, active fire hotspots, and meteorological "
        f"dispersion conditions (wind speed {weather_data.get('wind_speed_mps', 0):.1f} m/s, "
        f"humidity {weather_data.get('humidity_pct', 0):.0f}%)."
    )

    if fire_data.get("active_fire_count", 0) > 0:
        explanation += (
            f" {fire_data['active_fire_count']} active fire hotspot(s) were detected within "
            f"the search radius, consistent with the biomass-burning contribution."
        )

    recommendations = _build_recommendations(dominant_source, weather_data)

    return explanation, dominant_source, recommendations


def _build_recommendations(dominant_source: str, weather_data: Dict) -> List[str]:
    base = {
        "traffic": [
            "Consider traffic flow management / signal optimization in this corridor.",
            "Promote public transport or carpooling for commuters in this zone.",
        ],
        "construction": [
            "Enforce dust-control norms (water sprinkling, wind barriers) at active construction sites.",
            "Restrict high-dust construction activity during high-pollution hours.",
        ],
        "industries": [
            "Review industrial emission compliance for facilities within the search radius.",
            "Consider stricter stack-emission monitoring for nearby industrial units.",
        ],
        "biomass_burning": [
            "Coordinate with agricultural authorities on crop-residue burning alternatives.",
            "Issue public health advisories if biomass burning is expected to persist.",
        ],
    }
    recs = base.get(dominant_source, ["Investigate localized emission sources further."])

    if weather_data.get("wind_speed_mps", 5) < 1.5:
        recs.append("Low wind speed detected — pollutant accumulation risk is elevated.")

    return recs
