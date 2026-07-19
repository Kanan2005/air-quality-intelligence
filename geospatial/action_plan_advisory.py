"""Government action-plan + citizen risk-advisory generation via Groq's free LLM API.

Given a forecasted AQI and the surrounding context signals (traffic
congestion, land-use / industrial ratio, active-fire / biomass-burning
hotspots), asks the LLM to produce a structured, category-wise action
plan: what a municipal/pollution-control authority could plausibly do
right now, plus what has actually worked in other cities recently.

Tries several free-tier models in order, falling back to a conservative
static plan if Groq is unreachable or unconfigured -- the dashboard must
never show a blank panel.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import httpx
from logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

_SYSTEM_PROMPT = (
    "You are a senior environmental-policy advisor briefing a city pollution-control "
    "authority and its citizens during an air-quality event. You write concise, concrete, "
    "operationally realistic recommendations -- the kind an actual municipal task force "
    "could act on this week -- not generic platitudes. You draw on real interventions "
    "adopted by cities such as Delhi (GRAP), Beijing, Los Angeles, London (ULEZ), Seoul, "
    "and others where genuinely relevant. Respond with ONLY a single valid JSON object -- "
    "no markdown fences, no commentary before or after it."
)

_RESPONSE_SCHEMA_HINT = """Return ONLY this JSON shape:
{
  "citizen_advisory": ["...", "..."],
  "traffic_control": ["...", "..."],
  "vehicle_emission_control": ["...", "..."],
  "industrial_emission_control": ["...", "..."],
  "agri_biomass_control": ["...", "..."],
  "other_suggestions": ["...", "..."],
  "successful_city_models": [{"city": "...", "model": "...", "result": "..."}]
}
Rules:
- 2-4 bullet strings per list, each bullet a single actionable sentence, no numbering.
- "successful_city_models": 2-3 real, specific examples (city + program name + measured/claimed result).
- Tailor severity of recommendations to the AQI category given -- do not recommend emergency
  measures (odd-even, school closures) for Good/Satisfactory air quality.
- Ground recommendations in the context data provided (e.g. only emphasize agricultural
  biomass control if fire hotspot activity is non-trivial; only emphasize industrial control
  if industrial land ratio is meaningful; scale traffic recommendations to the congestion level given).
"""


def _build_prompt(
    aqi_value: int,
    category: str,
    dominant_pollutant: str,
    location_label: Optional[str],
    traffic_data: Optional[Dict],
    geo_data: Optional[Dict],
    fire_data: Optional[Dict],
) -> str:
    lines: List[str] = [
        f"Forecasted peak air quality: AQI {aqi_value} ({category}), dominant pollutant {dominant_pollutant}.",
    ]
    if location_label:
        lines.append(f"Location: {location_label}.")

    if traffic_data:
        lines.append(
            f"Traffic context: congestion level '{traffic_data.get('congestion_level', 'unknown')}' "
            f"({traffic_data.get('congestion_pct', 'unknown')}% below free-flow speed), "
            f"current speed {traffic_data.get('current_speed_kmph', 'unknown')} km/h."
        )
    if geo_data:
        lines.append(
            f"Land-use context: industrial land ratio {geo_data.get('industrial_area_ratio', 'unknown')}, "
            f"road density {geo_data.get('road_density_km_per_km2', 'unknown')} km/km², "
            f"construction sites nearby {geo_data.get('construction_site_count', 'unknown')}."
        )
    if fire_data:
        lines.append(
            f"Biomass burning context: {fire_data.get('active_fire_count', 'unknown')} active fire "
            f"hotspots detected nearby, mean intensity {fire_data.get('mean_frp_mw', 'n/a')} MW."
        )

    lines.append(
        "\nWrite a citizen risk advisory (short, practical, calm tone) AND a government/municipal "
        "action plan broken into the categories below. " + _RESPONSE_SCHEMA_HINT
    )
    return "\n".join(lines)


def _static_fallback_plan(aqi_value: int, category: str) -> Dict:
    severe = aqi_value >= 200
    return {
        "citizen_advisory": [
            "Limit prolonged outdoor exertion, especially for children, the elderly, and those with respiratory conditions."
            if severe else "Air quality is broadly acceptable; sensitive groups should still watch for symptoms.",
            "Keep windows closed during peak traffic hours and use an air purifier indoors if available."
            if severe else "Ventilate homes normally; no special precautions needed for the general public.",
            "Wear an N95 mask outdoors if you have a pre-existing respiratory or cardiac condition." if severe else "",
        ],
        "traffic_control": [
            "Stagger peak-hour traffic signals on congested arterial roads to reduce idling emissions.",
            "Deploy traffic police at high-congestion junctions during forecasted peak-AQI windows.",
        ],
        "vehicle_emission_control": [
            "Intensify roadside Pollution Under Control (PUC) certificate checks on high-traffic corridors.",
            "Restrict entry of older, non-compliant diesel commercial vehicles into the city core during the event.",
        ],
        "industrial_emission_control": [
            "Increase inspection frequency for industrial units in the affected zone and enforce emission-norm compliance.",
            "Temporarily curtail operations at units without adequate pollution-control equipment during severe events.",
        ],
        "agri_biomass_control": [
            "Coordinate with agriculture department to monitor and discourage crop-residue burning in the surrounding districts.",
            "Promote subsidized access to happy-seeder/crop-residue management machinery for local farmers.",
        ],
        "other_suggestions": [
            "Deploy mobile water-sprinklers and anti-smog guns on major dust-generating roads and construction sites.",
            "Issue public alerts via SMS/app notifications when forecasted AQI crosses unhealthy thresholds.",
        ],
        "successful_city_models": [
            {
                "city": "Delhi",
                "model": "Graded Response Action Plan (GRAP)",
                "result": "Pre-defined, AQI-triggered escalation of restrictions (construction bans, vehicle curbs) ahead of severe episodes.",
            },
            {
                "city": "Beijing",
                "model": "Coal-to-gas conversion + industrial relocation",
                "result": "Reported multi-year reduction in winter PM2.5 levels through fuel-switching and heavy-industry relocation.",
            },
            {
                "city": "London",
                "model": "Ultra Low Emission Zone (ULEZ)",
                "result": "Reported reduction in roadside NO2 concentrations within the zone after implementation.",
            },
        ],
        "source": "static_fallback",
    }


async def generate_action_plan(
    aqi_value: int,
    category: str,
    dominant_pollutant: str,
    location_label: Optional[str] = None,
    traffic_data: Optional[Dict] = None,
    geo_data: Optional[Dict] = None,
    fire_data: Optional[Dict] = None,
) -> Dict:
    """Generate a category-wise citizen advisory + government action plan."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set; using static fallback action plan.")
        return _static_fallback_plan(aqi_value, category)

    prompt = _build_prompt(
        aqi_value, category, dominant_pollutant, location_label, traffic_data, geo_data, fire_data
    )

    async with httpx.AsyncClient(timeout=20.0) as client:
        for model in MODEL_FALLBACK_CHAIN:
            try:
                resp = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 1000,
                        "response_format": {"type": "json_object"},
                    },
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning("Groq model %s unavailable (%s); trying next.", model, resp.status_code)
                    continue
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                for key in (
                    "citizen_advisory", "traffic_control", "vehicle_emission_control",
                    "industrial_emission_control", "agri_biomass_control",
                    "other_suggestions", "successful_city_models",
                ):
                    parsed.setdefault(key, [])
                parsed["source"] = f"groq_llm:{model}"
                logger.info("Action plan generated using model=%s (AQI=%s)", model, aqi_value)
                return parsed
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.warning("Groq model %s returned unparseable output (%s); trying next.", model, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq model %s failed (%s); trying next.", model, exc)
                continue

    logger.warning("All Groq models unavailable; using static fallback action plan.")
    return _static_fallback_plan(aqi_value, category)