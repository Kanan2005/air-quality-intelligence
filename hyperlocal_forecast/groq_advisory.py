"""Multilingual health-advisory generation via Groq's free LLM API.

Each requested language is generated as its own LLM call (run concurrently),
rather than asking one call to produce every language at once. Two reasons:

1. Token budget: non-Latin scripts (Hindi, Bengali, Tamil, Telugu, ...) cost
   far more tokens per character than English. A single shared max_tokens
   budget across several languages silently truncates the later ones.
2. Reliability: smaller fallback models are much worse at following a
   "produce N languages in this exact format" instruction than a plain
   "produce 1 language" instruction.

If an individual language's call fails, only that language falls back to a
static advisory -- the rest of the selected languages still render normally.
"""
from __future__ import annotations

import asyncio
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

# Per-language token budget. Non-Latin scripts need more tokens per word,
# so this is generous rather than tight -- a single language response is
# cheap even at this ceiling.
MAX_TOKENS_PER_LANGUAGE = 350


async def _generate_single_language(
    client: httpx.AsyncClient,
    aqi_value: int,
    category: str,
    dominant_pollutant: str,
    language: str,
) -> str:
    """Generate the advisory for exactly one language. Never raises -- falls
    back to a static advisory in that language's slot on any failure."""
    prompt = (
        f"Forecasted air quality: AQI {aqi_value} ({category}), dominant pollutant {dominant_pollutant}. "
        f"Write a short, practical citizen health advisory (max 80 words) with 2-3 concrete recommendations, "
        f"in {language}. Respond with ONLY the advisory text in {language} -- no heading, no preamble, "
        f"no English translation alongside it."
    )

    if not GROQ_API_KEY:
        return _fallback_text(aqi_value, category)

    for model in MODEL_FALLBACK_CHAIN:
        try:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": MAX_TOKENS_PER_LANGUAGE,
                },
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(
                    "Groq model %s unavailable (%s) for language=%s; trying next.",
                    model, resp.status_code, language,
                )
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text:
                logger.info("Groq advisory generated using model=%s language=%s", model, language)
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq model %s failed (%s) for language=%s; trying next.", model, exc, language)
            continue

    logger.warning("All Groq models unavailable for language=%s; using static fallback.", language)
    return _fallback_text(aqi_value, category)


async def generate_multilingual_advisory(
    aqi_value: int, category: str, dominant_pollutant: str, languages: List[str]
) -> str:
    """Generate the advisory in every requested language and join them under
    per-language headings, preserving the order the user selected."""
    if not languages:
        languages = ["English"]

    async with httpx.AsyncClient(timeout=20.0) as client:
        results = await asyncio.gather(
            *[
                _generate_single_language(client, aqi_value, category, dominant_pollutant, lang)
                for lang in languages
            ]
        )

    return "\n\n".join(f"### {lang}\n{text}" for lang, text in zip(languages, results))


def _fallback_text(aqi_value: int, category: str) -> str:
    return f"AQI is {aqi_value} ({category}). Limit outdoor exertion; sensitive groups should stay indoors."
