"""Multilingual health-advisory generation via Groq's free LLM API.
Tries several free-tier models in order, falling back if one is rate-limited or down.
"""
from __future__ import annotations
import os
from typing import List
import httpx
from logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ordered by preference; each is tried in turn on 429/5xx/timeout.
MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


async def generate_multilingual_advisory(
    aqi_value: int, category: str, dominant_pollutant: str, languages: List[str]
) -> str:
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set; using static fallback advisory.")
        return _fallback_advisory(aqi_value, category, languages)

    prompt = (
        f"Forecasted air quality: AQI {aqi_value} ({category}), dominant pollutant {dominant_pollutant}. "
        f"Write a short, practical citizen health advisory (max 80 words) with 2-3 concrete recommendations. "
        f"Provide it separately in each of these languages: {', '.join(languages)}. "
        f"Format as markdown with a '### <Language>' heading per language."
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in MODEL_FALLBACK_CHAIN:
            try:
                resp = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.4,
                        "max_tokens": 600,
                    },
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning("Groq model %s unavailable (%s); trying next.", model, resp.status_code)
                    continue
                resp.raise_for_status()
                logger.info("Groq advisory generated using model=%s", model)
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                logger.warning("Groq model %s failed (%s); trying next.", model, exc)
                continue

    logger.warning("All Groq models unavailable; using static fallback advisory.")
    return _fallback_advisory(aqi_value, category, languages)


def _fallback_advisory(aqi_value: int, category: str, languages: List[str]) -> str:
    base = f"AQI is {aqi_value} ({category}). Limit outdoor exertion; sensitive groups should stay indoors."
    return "\n\n".join(f"### {lang}\n{base}" for lang in languages)