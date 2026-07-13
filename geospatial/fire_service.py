"""
Active-fire / biomass-burning hotspot service using NASA FIRMS.

Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/
Falls back to synthetic hotspot data when unavailable.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Dict

try:
    from .config import settings
    from .mock_data import mock_fire_data
    from .http_client import get
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from mock_data import mock_fire_data
    from http_client import get
    from logger import get_logger

logger = get_logger(__name__)


async def fetch_fire_data(lat: float, lon: float, radius_km: float) -> Dict:
    """Fetch nearby active-fire hotspots (proxy for biomass/crop-residue burning)."""
    if not settings.NASA_FIRMS_MAP_KEY:
        logger.info("No NASA_FIRMS_MAP_KEY configured; using synthetic fire data.")
        return mock_fire_data(lat, lon, radius_km)

    try:
        # Bounding box: convert radius (km) to a rough lat/lon degree delta
        delta_deg = radius_km / 111.0
        bbox = f"{lon - delta_deg},{lat - delta_deg},{lon + delta_deg},{lat + delta_deg}"

        url = (
            f"{settings.NASA_FIRMS_BASE_URL}/area/csv/"
            f"{settings.NASA_FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{bbox}/1"
        )

        logger.debug("Fetching NASA FIRMS hotspot data for (%s, %s), bbox=%s", lat, lon, bbox)
        resp = await get(url, timeout=settings.HTTP_TIMEOUT_SECONDS)
        text = resp.text

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.info(
                "Fire hotspot data fetched from NASA FIRMS for (%s, %s) — no active hotspots in range",
                lat, lon,
            )
            return {
                "active_fire_count": 0,
                "nearest_fire_distance_km": None,
                "mean_frp_mw": None,
                "source": "nasa_firms_live",
            }

        distances = []
        frps = []
        for row in rows:
            try:
                r_lat, r_lon = float(row["latitude"]), float(row["longitude"])
                distances.append(_haversine(lat, lon, r_lat, r_lon))
                frps.append(float(row.get("frp", 0)))
            except (KeyError, ValueError):
                continue

        logger.info("Fire hotspot data fetched from NASA FIRMS for (%s, %s)", lat, lon)
        return {
            "active_fire_count": len(rows),
            "nearest_fire_distance_km": round(min(distances), 2) if distances else None,
            "mean_frp_mw": round(sum(frps) / len(frps), 1) if frps else None,
            "source": "nasa_firms_live",
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "NASA FIRMS fetch failed (%s: %s). Falling back to synthetic fire data.",
            type(exc).__name__, exc,
        )
        return mock_fire_data(lat, lon, radius_km)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))