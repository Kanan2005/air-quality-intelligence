"""
Pollution-source history service.

Uses Groq's free-tier LLM API to produce a short, structured historical
brief on major industrial sites and agricultural biomass-burning (crop
residue / stubble) incidents reported near a coordinate, for display in
the Geospatial Source Attribution tab.

This is explicitly LLM-recalled background context, not a verified
government record -- it is clearly labeled as such in the UI so it reads
as "useful investigative context for an enforcement officer to verify",
not as an authoritative source register. Falls back to a static,
non-hallucinated notice if Groq is unavailable.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import httpx

try:
    from .config import settings
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
GROQ_URL = settings.GROQ_CHAT_URL

# Ordered by preference; each is tried in turn on 429/5xx/timeout.
MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

_SYSTEM_PROMPT = (
    "You are an environmental-compliance research assistant helping a city "
    "pollution-control officer understand the industrial and agricultural "
    "burning history of a location before a field inspection. Respond with "
    "ONLY a single valid JSON object -- no markdown fences, no commentary "
    "before or after it."
)


def _build_prompt(lat: float, lon: float, region_hint: Optional[str], geo_data: Optional[Dict], fire_data: Optional[Dict]) -> str:
    context_bits = []
    if region_hint:
        context_bits.append(f"Approximate region/city: {region_hint}.")
    if geo_data:
        context_bits.append(
            f"Nearby land use signal: industrial land ratio "
            f"{geo_data.get('industrial_area_ratio', 'unknown')}, "
            f"road density {geo_data.get('road_density_km_per_km2', 'unknown')} km/km²."
        )
    if fire_data:
        context_bits.append(
            f"Recent active-fire hotspot signal: {fire_data.get('active_fire_count', 'unknown')} "
            f"detections nearby, dominant season signal only."
        )
    context_str = " ".join(context_bits)

    return (
        f"Coordinates: latitude {lat}, longitude {lon}. {context_str}\n\n"
        "Based on your general knowledge of this region of India (or the world, if outside "
        "India), summarize:\n"
        "1. Notable industrial sites, industrial clusters, or manufacturing belts historically "
        "known to operate near this location (name the industry type/sector; only name a "
        "specific facility if you are reasonably confident, otherwise describe the sector "
        "generically, e.g. 'brick kiln cluster', 'thermal power belt', 'textile dyeing units').\n"
        "2. Known agricultural biomass / crop-residue (stubble) burning patterns associated with "
        "this region -- typical season/months, crop type driving it (e.g. paddy stubble, sugarcane "
        "trash), and any notable past incidents or spikes reported in media/government data.\n"
        "3. A one-line overall summary of how much this location's air quality is plausibly "
        "shaped by industrial vs. agricultural biomass sources historically.\n\n"
        "Return ONLY this JSON shape:\n"
        "{\n"
        '  "industrial_history": [{"site_or_sector": "...", "period": "...", "note": "..."}],\n'
        '  "biomass_burning_history": [{"season_or_period": "...", "note": "..."}],\n'
        '  "summary": "...",\n'
        '  "confidence": "low|medium|high"\n'
        "}\n"
        "Use at most 4 items per list. If you have no reliable knowledge for this exact spot, "
        "generalize honestly to the district/state level and say so in the note fields rather "
        "than inventing a specific facility name."
    )


def _fallback_history(lat: float, lon: float) -> Dict:
    return {
        "industrial_history": [],
        "biomass_burning_history": [],
        "summary": (
            "AI-generated historical context is unavailable right now "
            "(no GROQ_API_KEY configured, or the Groq API is unreachable). "
            "Configure GROQ_API_KEY to enable this panel."
        ),
        "confidence": "unavailable",
        "source": "unavailable",
    }


async def generate_pollution_source_history(
    lat: float,
    lon: float,
    region_hint: Optional[str] = None,
    geo_data: Optional[Dict] = None,
    fire_data: Optional[Dict] = None,
) -> Dict:
    """Return an LLM-generated brief on industrial + biomass-burning history near (lat, lon). I just have traffic data, fetch and analyze the industrial sites and geospatial history as much as you can."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set; skipping pollution source history generation.")
        return _fallback_history(lat, lon)

    prompt = _build_prompt(lat, lon, region_hint, geo_data, fire_data)

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
                        "temperature": 0.3,
                        "max_tokens": 700,
                        "response_format": {"type": "json_object"},
                    },
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning("Groq model %s unavailable (%s); trying next.", model, resp.status_code)
                    continue
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                parsed.setdefault("industrial_history", [])
                parsed.setdefault("biomass_burning_history", [])
                parsed.setdefault("summary", "")
                parsed.setdefault("confidence", "medium")
                parsed["source"] = f"groq_llm:{model}"
                logger.info("Pollution source history generated using model=%s for (%s, %s)", model, lat, lon)
                return parsed
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.warning("Groq model %s returned unparseable output (%s); trying next.", model, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq model %s failed (%s); trying next.", model, exc)
                continue

    logger.warning("All Groq models unavailable; returning fallback pollution source history.")
    return _fallback_history(lat, lon)