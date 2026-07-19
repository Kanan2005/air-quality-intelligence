"""
Traffic congestion service using TomTom's Traffic API (Flow Segment Data).

Requires a free API key from https://developer.tomtom.com/ (TOMTOM_API_KEY).
Falls back to synthetic congestion data when unavailable, mirroring the
pattern used by weather_service.py / fire_service.py.

TomTom's Flow Segment Data endpoint returns the current and "free flow"
(uncongested) speed/travel-time for the road segment nearest a point. We
derive a normalized congestion ratio from that, which is a much closer
proxy for "traffic pollution pressure" at a coordinate than raw speed
alone, since a 20 km/h reading means very different things on a highway
vs. a residential lane.
"""
from __future__ import annotations

from typing import Dict

try:
    from .config import settings
    from .mock_data import mock_traffic
    from .http_client import get
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from mock_data import mock_traffic
    from http_client import get
    from logger import get_logger

logger = get_logger(__name__)


def _congestion_level(ratio: float) -> str:
    if ratio < 0.10:
        return "Free Flow"
    if ratio < 0.30:
        return "Light"
    if ratio < 0.55:
        return "Moderate"
    if ratio < 0.75:
        return "Heavy"
    return "Severe"


async def fetch_traffic_congestion(lat: float, lon: float, zoom: int = 10) -> Dict:
    """Fetch live traffic flow / congestion for the road segment nearest (lat, lon).

    `zoom` controls the TomTom tile zoom level used to pick the segment
    (10 is a reasonable default for city-scale queries; higher values
    narrow in on smaller roads).
    """
    if not settings.TOMTOM_API_KEY:
        logger.info("No TOMTOM_API_KEY configured; using synthetic traffic data.")
        return mock_traffic(lat, lon)

    try:
        url = f"{settings.TOMTOM_TRAFFIC_BASE_URL}/flowSegmentData/absolute/{zoom}/json"
        params = {
            "point": f"{lat},{lon}",
            "key": settings.TOMTOM_API_KEY,
            "unit": "KMPH",
        }
        resp = await get(url, params=params, timeout=settings.TOMTOM_TIMEOUT_SECONDS)
        payload = resp.json().get("flowSegmentData", {})

        current_speed = payload.get("currentSpeed")
        free_flow_speed = payload.get("freeFlowSpeed")
        current_travel_time = payload.get("currentTravelTime")
        free_flow_travel_time = payload.get("freeFlowTravelTime")
        confidence = payload.get("confidence")
        road_closure = payload.get("roadClosure", False)
        frc = payload.get("frc")

        if free_flow_speed and free_flow_speed > 0 and current_speed is not None:
            congestion_ratio = max(0.0, 1 - (current_speed / free_flow_speed))
        elif free_flow_travel_time and free_flow_travel_time > 0 and current_travel_time is not None:
            congestion_ratio = max(0.0, (current_travel_time / free_flow_travel_time) - 1)
        else:
            raise ValueError("TomTom response missing usable speed/travel-time fields")
        congestion_ratio = round(min(congestion_ratio, 1.0), 3)

        logger.info("Traffic congestion fetched from TomTom (live) for (%s, %s)", lat, lon)
        return {
            "current_speed_kmph": current_speed,
            "free_flow_speed_kmph": free_flow_speed,
            "current_travel_time_s": current_travel_time,
            "free_flow_travel_time_s": free_flow_travel_time,
            "congestion_ratio": congestion_ratio,
            "congestion_pct": round(congestion_ratio * 100, 1),
            "congestion_level": _congestion_level(congestion_ratio),
            "confidence": confidence,
            "road_closure": bool(road_closure),
            "road_class": frc,
            "source": "tomtom_live",
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TomTom traffic fetch failed (%s: %s). Falling back to synthetic traffic data.",
            type(exc).__name__, exc,
        )
        return mock_traffic(lat, lon)