"""
Geospatial context service.

Queries OpenStreetMap's Overpass API for road network density, industrial
land-use, and construction sites around a coordinate. Falls back to
deterministic synthetic land-cover data (approximating ESA WorldCover style
ratios) when the live query fails.
"""
from __future__ import annotations

import math
from typing import Dict

try:
    from .config import settings
    from .mock_data import mock_geospatial
    from .http_client import post
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from mock_data import mock_geospatial
    from http_client import post
    from logger import get_logger

logger = get_logger(__name__)

# Overpass's public instance enforces a usage policy requiring an
# identifiable client; requests without a User-Agent are rejected with
# 406 Not Acceptable before the query is even evaluated.
_HEADERS = {"User-Agent": "geospatial-pollution-agent/1.0 (india-runs-ideathon)"}

# Multiple public Overpass mirrors. overpass-api.de is the canonical instance
# but is frequently overloaded (504s under load). Falling through several
# independently-operated mirrors materially improves success rate.
# NOTE: settings.OVERPASS_BASE_URL / OVERPASS_MIRROR_URL are tried first (in
# case you've configured a private/paid instance), then these known-good
# public fallbacks.
_FALLBACK_OVERPASS_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",  
]


def _candidate_urls() -> list[str]:
    configured = [
        u for u in (
            getattr(settings, "OVERPASS_BASE_URL", None),
            getattr(settings, "OVERPASS_MIRROR_URL", None),
        )
        if u
    ]
    # De-dupe while preserving order, then append fallbacks not already present.
    seen = set()
    ordered = []
    for u in configured + _FALLBACK_OVERPASS_URLS:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _overpass_query(lat: float, lon: float, radius_m: int, server_timeout: int, lightweight: bool = False) -> str:
    """Build the Overpass QL query.

    lightweight=True drops the `geom` output modifier for road ways on later
    retry attempts. Geometry serialization is the most expensive part of the
    response for busy urban areas and is a common cause of 504s on loaded
    mirrors; road length falls back to a density estimate from way count
    instead of haversine summation when this is used.
    """
    out_clause = "out tags geom;" if not lightweight else "out tags;"
    return f"""
    [out:json][timeout:{server_timeout}];
    (
      way["highway"](around:{radius_m},{lat},{lon});
      way["landuse"="industrial"](around:{radius_m},{lat},{lon});
      way["landuse"="construction"](around:{radius_m},{lat},{lon});
      node["construction"](around:{radius_m},{lat},{lon});
    );
    {out_clause}
    """


async def fetch_geospatial_features(lat: float, lon: float, radius_km: float) -> Dict:
    """Fetch OSM-derived road density and land-use ratios around a point."""
    radius_m = int(radius_km * 1000)
    urls = _candidate_urls()

    last_exc: Exception | None = None

    for attempt, url in enumerate(urls):
        is_last_attempt = attempt == len(urls) - 1
        # Give later attempts a shorter server-side timeout and a lighter
        # payload, so we fail fast and don't stack up client-side wait time
        # across 3-4 sequential mirror attempts.
        lightweight = attempt > 0
        server_timeout = max(10, settings.OVERPASS_SERVER_TIMEOUT_SECONDS - attempt * 5)
        client_timeout = server_timeout + 5  # client timeout MUST exceed server-side [timeout:] or you'll cut off a response the server was about to send

        query = _overpass_query(lat, lon, radius_m, server_timeout, lightweight=lightweight)

        try:
            logger.debug("Querying Overpass (%s, attempt %d) for (%s, %s)", url, attempt, lat, lon)
            resp = await post(
                url,
                data={"data": query},
                headers=_HEADERS,
                timeout=client_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            elements = data.get("elements", [])
            if not elements:
                raise ValueError("No OSM elements returned for this area")

            result = _parse_osm_elements(elements, radius_km)
            logger.info("Geospatial features fetched from OSM/Overpass (%s) for (%s, %s)", url, lat, lon)
            return result

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not is_last_attempt:
                logger.warning(
                    "OSM/Overpass fetch from %s failed (%s: %s). Trying next mirror.",
                    url, type(exc).__name__, exc,
                )
                continue
            logger.warning(
                "OSM/Overpass fetch failed (%s: %s) on all %d instance(s) tried. Falling back to synthetic geospatial data.",
                type(exc).__name__, exc, len(urls),
            )
            return mock_geospatial(lat, lon, radius_km)

    # Unreachable in practice (urls is never empty given fallback list), but
    # keeps type-checkers happy and guards against a future empty-list bug.
    logger.warning("No Overpass URLs configured or reachable (%s). Falling back to synthetic geospatial data.", last_exc)
    return mock_geospatial(lat, lon, radius_km)


def _haversine_length_km(coords: list) -> float:
    """Approximate the length of a way (list of {lat, lon} dicts) in km."""
    total = 0.0
    for i in range(1, len(coords)):
        total += _haversine(
            coords[i - 1]["lat"], coords[i - 1]["lon"], coords[i]["lat"], coords[i]["lon"]
        )
    return total


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_osm_elements(elements: list, radius_km: float) -> Dict:
    road_len_km = 0.0
    major_roads = 0
    industrial_ways = 0
    construction_count = 0

    major_highway_types = {"motorway", "trunk", "primary", "secondary"}

    # Rough average length (km) per OSM "way" segment, used to estimate road
    # density when geometry wasn't requested (lightweight retry query).
    _AVG_WAY_LENGTH_KM = 0.3

    for el in elements:
        tags = el.get("tags", {})
        if "highway" in tags:
            if el.get("geometry"):
                road_len_km += _haversine_length_km(el["geometry"])
            else:
                road_len_km += _AVG_WAY_LENGTH_KM
            if tags["highway"] in major_highway_types:
                major_roads += 1
        if tags.get("landuse") == "industrial":
            industrial_ways += 1
        if tags.get("landuse") == "construction" or "construction" in tags:
            construction_count += 1

    area_km2 = math.pi * (radius_km ** 2)
    road_density = road_len_km / area_km2 if area_km2 > 0 else 0.0
    industrial_ratio = min(0.9, industrial_ways / max(1, len(elements)) * 3)
    green_cover = max(0.02, 1 - industrial_ratio - min(0.7, road_density / 20))
    built_up = min(0.95, industrial_ratio + road_density / 15)

    return {
        "road_density_km_per_km2": round(road_density, 2),
        "major_road_count": major_roads,
        "industrial_area_ratio": round(industrial_ratio, 3),
        "construction_site_count": float(construction_count),
        "green_cover_ratio": round(green_cover, 3),
        "built_up_ratio": round(built_up, 3),
        "source": "osm_overpass_live",
    }