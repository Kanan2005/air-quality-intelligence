"""Open-Meteo Air Quality forecast client — free, keyless, global coverage."""
from __future__ import annotations
from typing import Dict
import httpx
from logger import get_logger

logger = get_logger(__name__)

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_FIELDS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi"


async def fetch_hyperlocal_forecast(lat: float, lon: float, forecast_days: int = 5) -> Dict:
    """Hourly pollutant + AQI forecast for any point on Earth. No API key required."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": HOURLY_FIELDS, "forecast_days": forecast_days, "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data.get("hourly", {})
    n_hours = len(hourly.get("time", []))
    pm25_missing = sum(1 for v in hourly.get("pm2_5", []) if v is None)
    pm10_missing = sum(1 for v in hourly.get("pm10", []) if v is None)
    logger.info(
        "Open-Meteo forecast fetched for (%s, %s): %d hours, pm2_5 missing=%d, pm10 missing=%d",
        lat, lon, n_hours, pm25_missing, pm10_missing,
    )
    return hourly