"""Weather data acquisition service (OpenWeatherMap, with mock fallback)."""
from __future__ import annotations

from typing import Dict

from config import settings
from mock_data import mock_weather
from http_client import get_json
from logger import get_logger

logger = get_logger(__name__)


async def fetch_weather(lat: float, lon: float) -> Dict:
    """Fetch current weather conditions, used to reason about pollutant dispersion."""
    if not settings.OPENWEATHER_API_KEY:
        logger.info("No OPENWEATHER_API_KEY configured; using synthetic weather data.")
        return mock_weather(lat, lon)

    try:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": settings.OPENWEATHER_API_KEY,
            "units": "metric",
        }
        data = await get_json(
            f"{settings.OPENWEATHER_BASE_URL}/weather",
            params=params,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
        )

        logger.info("Weather data fetched from OpenWeatherMap for (%s, %s)", lat, lon)
        return {
            "temperature_c": round(data["main"]["temp"], 1),
            "humidity_pct": round(data["main"]["humidity"], 1),
            "wind_speed_mps": round(data.get("wind", {}).get("speed", 0.0), 2),
            "wind_direction_deg": round(data.get("wind", {}).get("deg", 0.0), 1),
            "pressure_hpa": round(data["main"]["pressure"], 1),
            "source": "openweather_live",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OpenWeatherMap fetch failed (%s). Falling back to synthetic weather data.", exc
        )
        return mock_weather(lat, lon)