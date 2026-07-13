"""Air Quality Index data acquisition service (CPCB, with mock fallback)."""
from __future__ import annotations

from typing import Dict

try:
    from .config import settings
    from .mock_data import mock_aqi
    from .cpcb_service import fetch_cpcb_aqi
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from mock_data import mock_aqi
    from cpcb_service import fetch_cpcb_aqi
    from logger import get_logger

logger = get_logger(__name__)

async def fetch_aqi(lat: float, lon: float, radius_km: float = 5.0) -> Dict:
    if not settings.CPCB_API_KEY:
        logger.warning("CPCB_API_KEY not set. Falling back to synthetic AQI data.")
        return mock_aqi(lat, lon)

    try:
        result = await fetch_cpcb_aqi(lat, lon, radius_km)
        logger.info(
            "aqi_service: using CPCB data for (%s, %s) — station %.1f km away, fallback=%s",
            lat, lon, result.get("station_distance_km", -1), result.get("is_distant_fallback", False),
        )
        return result
    except Exception as exc:
        logger.warning("CPCB fetch failed (%s: %s). Falling back to synthetic AQI data.", type(exc).__name__, exc)
        return mock_aqi(lat, lon)