"""
Air Quality data acquisition service for CPCB (Central Pollution Control
Board, India), via the data.gov.in open-data proxy.

CPCB does not expose a lat/lon-radius search endpoint, and pulling the
*entire* national snapshot (thousands of rows) on every call is too slow
for data.gov.in to serve within a reasonable timeout. So instead:

1. Reverse-geocode (lat, lon) to an Indian state name via OSM Nominatim.
2. Ask data.gov.in to filter server-side to that state (`filters[state]`),
   which cuts the payload from ~5000 rows to a couple hundred.
3. Among the returned stations, compute distance to (lat, lon) and keep
   the nearest one that reports PM2.5.

If reverse-geocoding fails or the state filter comes back empty, we fall
back once to an unfiltered full-snapshot pull as a last resort before
giving up and letting the caller fall back to mock data.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import httpx

try:
    from .config import settings
    from .mock_data import _aqi_category, _pm25_to_aqi
    from .http_client import get_json
    from .logger import get_logger
except ImportError:  # Fallback for direct execution from the geospatial folder
    from config import settings
    from mock_data import _aqi_category, _pm25_to_aqi
    from http_client import get_json
    from logger import get_logger

logger = get_logger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_USER_AGENT = "geospatial-pollution-agent/1.0 (india-runs-ideathon)"

# data.gov.in appears to silently drop (rather than cleanly reject)
# requests carrying the default httpx/requests library User-Agent -- this
# looks identical to a network timeout. Sending a normal browser-style
# Accept header + a non-library User-Agent avoids that.
_DATA_GOV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# In-memory cache of CPCB rows, keyed by state name ("__ALL__" for the
# unfiltered fallback). Readings only change hourly-ish, so we avoid
# re-fetching on every /analyze call within the TTL window.
_cache: Dict[str, Dict[str, object]] = {}

# CPCB/data.gov.in pollutant_id values -> our internal field names.
# Field names on data.gov.in have been observed to vary in casing/spacing
# across resource revisions, so we normalize before matching.
_POLLUTANT_MAP = {
    "pm2.5": "pm25",
    "pm25": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "co": "co",
    "ozone": "o3",
    "o3": "o3",
    "nh3": "nh3",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _get(record: dict, *keys: str) -> Optional[str]:
    """Case/format-tolerant field lookup across known data.gov.in key variants."""
    for key in keys:
        for candidate in (key, key.lower(), key.upper(), key.replace("_", " ")):
            if candidate in record and record[candidate] not in ("", "NA", None):
                return record[candidate]
    return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _reverse_geocode_state(lat: float, lon: float) -> Optional[str]:
    """Best-effort reverse geocode to an Indian state name via Nominatim."""
    try:
        params = {"lat": lat, "lon": lon, "format": "jsonv2"}
        data = await get_json(_NOMINATIM_URL, params=params, headers={"User-Agent": _USER_AGENT}, timeout=8.0)
        state = (data.get("address") or {}).get("state")
        return state
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reverse geocode for state failed (%s: %s)", type(exc).__name__, exc)
        return None


async def _fetch_records(*, state: Optional[str], limit: int, timeout: float) -> List[dict]:
    cache_key = state or "__ALL__"
    now = time.monotonic()
    entry = _cache.get(cache_key)
    if entry and (now - entry["fetched_at"]) < settings.CPCB_CACHE_TTL_SECONDS:
        logger.debug("Using cached CPCB records for '%s' (%d rows)", cache_key, len(entry["records"]))
        return entry["records"]

    url = f"{settings.CPCB_BASE_URL}/{settings.CPCB_RESOURCE_ID}"
    params = {
        "api-key": settings.CPCB_API_KEY,
        "format": "json",
        "limit": limit,
    }
    if state:
        params["filters[state]"] = state

    payload = await get_json(url, params=params, headers=_DATA_GOV_HEADERS, timeout=timeout)

    records = payload.get("records", [])
    if not records:
        raise ValueError(f"CPCB/data.gov.in returned no records for state='{state}'")

    _cache[cache_key] = {"records": records, "fetched_at": now}
    return records


async def _fetch_all_records(lat: float, lon: float) -> List[dict]:
    if not settings.CPCB_API_KEY:
        raise ValueError("CPCB_API_KEY is not set")

    state = await _reverse_geocode_state(lat, lon)
    if state:
        try:
            return await _fetch_records(state=state, limit=1000, timeout=settings.CPCB_TIMEOUT_SECONDS)
        except ValueError as exc:
            # Empty/no-match result for this state -> worth trying the
            # unfiltered pull, it might just be a filter-name mismatch.
            logger.warning(
                "CPCB state-filtered fetch for '%s' returned no data (%s). Trying full snapshot.",
                state, exc,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # A timeout/connection failure will almost certainly happen
            # again on a *bigger* unfiltered request -- don't double the
            # wait for a near-certain repeat failure.
            logger.warning(
                "CPCB state-filtered fetch for '%s' failed (%s: %s) -- likely a "
                "connectivity issue to data.gov.in, not a payload-size issue. "
                "Skipping the full-snapshot retry.",
                state, type(exc).__name__, exc,
            )
            raise

    # Last resort: unfiltered full-country pull (slow, only reached if
    # geocoding or the state filter didn't work out).
    return await _fetch_records(state=None, limit=settings.CPCB_FETCH_LIMIT, timeout=settings.CPCB_TIMEOUT_SECONDS)


def _nearest_station_records(records, lat, lon, radius_km, max_fallback_km=30):
    by_station, station_distance = {}, {}
    for rec in records:
        s_lat = _to_float(_get(rec, "latitude"))
        s_lon = _to_float(_get(rec, "longitude"))
        station = _get(rec, "station") or _get(rec, "station_name")
        if s_lat is None or s_lon is None or not station:
            continue
        dist = _haversine_km(lat, lon, s_lat, s_lon)
        by_station.setdefault(station, []).append(rec)
        station_distance[station] = min(station_distance.get(station, dist), dist)

    if not by_station:
        raise ValueError("No CPCB stations found in this state's dataset")

    ranked = sorted(station_distance.items(), key=lambda kv: kv[1])

    # Pass 1: strictly within the user's requested radius, prefer PM2.5-reporting station
    within_radius = [s for s, d in ranked if d <= radius_km]
    for station in within_radius:
        if any(_normalize_pollutant(r) == "pm25" for r in by_station[station]):
            return by_station[station], station_distance[station], False  # (rows, dist, is_fallback)

    # Pass 2: nothing usable within the requested radius -> widen the search to
    # every station we fetched (state-wide), still preferring one that reports
    # PM2.5. This is the key fix: the single geographically-nearest station
    # (picked by distance alone, ignoring what it measures) often turns out to
    # only report PM10/NO2/SO2/etc, which used to make the whole lookup fail
    # even when a usable PM2.5 station existed slightly farther away.
    for station, dist in ranked:
        if any(_normalize_pollutant(r) == "pm25" for r in by_station[station]):
            is_fallback = dist > max(radius_km, max_fallback_km)
            logger.info(
                "No PM2.5 station within %.1f km; using nearest PM2.5-reporting "
                "station '%s' %.1f km away instead.",
                radius_km, station, dist,
            )
            return by_station[station], dist, is_fallback

    # Pass 3: true last resort -- nothing in the entire fetched dataset reports
    # PM2.5 at all. Take the nearest station regardless of pollutant; the
    # caller (_parse_station_rows) will raise if it truly has no usable
    # reading, and aqi_service will fall back to synthetic data from there.
    nearest_station, nearest_dist = ranked[0]
    is_fallback = nearest_dist > max(radius_km, max_fallback_km)
    logger.warning(
        "No station in this state's dataset reports PM2.5; falling back to "
        "nearest station '%s' %.1f km away (may lack a usable PM2.5 reading).",
        nearest_station, nearest_dist,
    )
    return by_station[nearest_station], nearest_dist, is_fallback


def _normalize_pollutant(rec: dict) -> Optional[str]:
    raw = _get(rec, "pollutant_id", "pollutant_ID", "pollutant")
    if raw is None:
        return None
    return _POLLUTANT_MAP.get(str(raw).strip().lower())


def _pollutant_value(rec: dict) -> Optional[float]:
    avg = _to_float(_get(rec, "pollutant_avg", "avg_value"))
    if avg is not None:
        return avg
    lo = _to_float(_get(rec, "pollutant_min", "min_value"))
    hi = _to_float(_get(rec, "pollutant_max", "max_value"))
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    return lo if lo is not None else hi


def _parse_station_rows(rows: List[dict]) -> Optional[Dict]:
    values: Dict[str, float] = {}
    for rec in rows:
        key = _normalize_pollutant(rec)
        val = _pollutant_value(rec)
        if key and val is not None and key not in values:
            values[key] = val

    pm25 = values.get("pm25")
    if pm25 is None:
        return None

    aqi = _pm25_to_aqi(pm25)
    return {
        "aqi": aqi,
        "pm25": round(pm25, 1),
        "pm10": round(values.get("pm10", pm25 * 1.6), 1),
        "no2": round(values["no2"], 1) if "no2" in values else None,
        "so2": round(values["so2"], 1) if "so2" in values else None,
        "co": round(values["co"], 2) if "co" in values else None,
        "o3": round(values["o3"], 1) if "o3" in values else None,
        "category": _aqi_category(aqi),
        "source": "cpcb_live",
    }


async def fetch_cpcb_aqi(lat, lon, radius_km=5.0):
    records = await _fetch_all_records(lat, lon)
    rows, distance_km, is_fallback = _nearest_station_records(records, lat, lon, radius_km)
    parsed = _parse_station_rows(rows)
    if parsed is None:
        raise ValueError("Nearest CPCB station found but no usable PM2.5 reading")
    parsed["station_distance_km"] = round(distance_km, 1)
    parsed["is_distant_fallback"] = is_fallback
    return parsed