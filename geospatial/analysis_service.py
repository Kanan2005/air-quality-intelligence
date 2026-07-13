"""Orchestrates the full source-attribution analysis pipeline for a single request."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict

from sqlalchemy.orm import Session

from app.ai.explainability import build_explanation, compute_confidence_score
from app.ai.feature_engineering import build_feature_vector, compute_data_quality_flags
from app.ai.source_attribution import get_model
from app.db.models import AnalysisRecord
from app.services.aqi_service import fetch_aqi
from app.services.fire_service import fetch_fire_data
from app.services.geospatial_service import fetch_geospatial_features
from app.services.weather_service import fetch_weather
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_analysis(
    lat: float, lon: float, radius_km: float, location_name: str | None, db: Session
) -> Dict:
    """Fetch all data sources concurrently, run inference, persist, and return a full result."""
    request_id = str(uuid.uuid4())
    logger.info("Starting analysis %s for (%s, %s), radius=%skm", request_id, lat, lon, radius_km)

    aqi_data, weather_data, geo_data, fire_data = await asyncio.gather(
        fetch_aqi(lat, lon, radius_km),
        fetch_weather(lat, lon),
        fetch_geospatial_features(lat, lon, radius_km),
        fetch_fire_data(lat, lon, radius_km),
    )

    feature_vector = build_feature_vector(aqi_data, weather_data, geo_data, fire_data)
    model = get_model()
    source_contribution, tree_agreement = model.predict(feature_vector)

    data_quality_flags = compute_data_quality_flags(aqi_data, weather_data, geo_data, fire_data)
    confidence_score = compute_confidence_score(tree_agreement, data_quality_flags, aqi_data)

    explanation, dominant_source, recommendations = build_explanation(
        source_contribution, aqi_data, weather_data, geo_data, fire_data
    )

    result = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc),
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "location_name": location_name,
        "aqi_data": aqi_data,
        "weather_data": weather_data,
        "geospatial_features": geo_data,
        "fire_data": fire_data,
        "source_contribution": source_contribution,
        "confidence_score": confidence_score,
        "explanation": explanation,
        "dominant_source": dominant_source,
        "recommendations": recommendations,
        "data_quality_flags": data_quality_flags,
    }

    _persist_analysis(db, result)
    logger.info("Analysis %s complete. Dominant source=%s confidence=%s", request_id, dominant_source, confidence_score)

    return result


def _persist_analysis(db: Session, result: Dict) -> None:
    try:
        record = AnalysisRecord(
            request_id=result["request_id"],
            timestamp=result["timestamp"],
            latitude=result["latitude"],
            longitude=result["longitude"],
            radius_km=result["radius_km"],
            location_name=result.get("location_name"),
            aqi=result["aqi_data"]["aqi"],
            pm25=result["aqi_data"]["pm25"],
            pm10=result["aqi_data"]["pm10"],
            traffic_pct=result["source_contribution"]["traffic"],
            construction_pct=result["source_contribution"]["construction"],
            industries_pct=result["source_contribution"]["industries"],
            biomass_burning_pct=result["source_contribution"]["biomass_burning"],
            dominant_source=result["dominant_source"],
            confidence_score=result["confidence_score"],
            explanation=result["explanation"],
            raw_payload={
                "weather": result["weather_data"],
                "geospatial": result["geospatial_features"],
                "fire": result["fire_data"],
                "data_quality_flags": result["data_quality_flags"],
            },
        )
        db.add(record)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist analysis record: %s", exc)
        db.rollback()
