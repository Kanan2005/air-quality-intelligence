"""Compatibility wrapper for the government AQI client."""

from govt_aqi_client import GovernmentAQIClient, INDIAN_CITIES, pm25_to_aqi

OpenAQClient = GovernmentAQIClient

__all__ = ["GovernmentAQIClient", "OpenAQClient", "INDIAN_CITIES", "pm25_to_aqi"]
