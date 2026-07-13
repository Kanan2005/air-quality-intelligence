"""
Central application configuration.

All values can be overridden via environment variables or a `.env` file.
See `.env.example` for the full list of supported settings.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General ---
    APP_NAME: str = "Geospatial Pollution Source Attribution Agent"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./data/pollution_agent.db"

    # --- External API keys (all optional -> falls back to mock data) ---
    OPENWEATHER_API_KEY: Optional[str] = None
    NASA_FIRMS_MAP_KEY: Optional[str] = None
    CPCB_API_KEY: Optional[str] = None

    # --- External API base URLs ---
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    OVERPASS_BASE_URL: str = "https://overpass-api.de/api/interpreter"
    # Public Overpass mirror, used as a retry target if the primary instance
    # times out (overpass-api.de is a shared free instance and gets
    # overloaded/queued during peak hours).
    OVERPASS_MIRROR_URL: str = "https://overpass.kumi.systems/api/interpreter"
    # Overpass road/landuse queries routinely need 15-30s on the public
    # instance -- this is decoupled from HTTP_TIMEOUT_SECONDS (6s), which is
    # tuned for fast APIs like OpenWeatherMap and is too tight for Overpass.
    OVERPASS_SERVER_TIMEOUT_SECONDS: int = 20  # sent as [timeout:N] in the query itself
    OVERPASS_CLIENT_TIMEOUT_SECONDS: float = 28.0  # must exceed the server timeout with margin
    NASA_FIRMS_BASE_URL: str = "https://firms.modaps.eosdis.nasa.gov/api"

    # --- CPCB (via data.gov.in) ---
    CPCB_BASE_URL: str = "https://api.data.gov.in/resource"
    CPCB_RESOURCE_ID: str = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    CPCB_FETCH_LIMIT: int = 5000  # data.gov.in paginates; this covers all live stations in one call
    CPCB_TIMEOUT_SECONDS: float = 25.0  # data.gov.in is slow for large record pulls
    CPCB_CACHE_TTL_SECONDS: int = 300  # reuse the full station snapshot for 5 min between calls

    # --- Behaviour toggles ---
    USE_MOCK_DATA_ON_FAILURE: bool = True
    HTTP_TIMEOUT_SECONDS: float = 6.0
    DEFAULT_SEARCH_RADIUS_KM: float = 3.0

    # --- Model ---
    MODEL_PATH: str = "geospatial/ai/artifacts/source_attribution_model.joblib"
    RANDOM_SEED: int = 42


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Convenience module-level singleton
settings = get_settings()

# Ensure data directory exists for SQLite file
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)