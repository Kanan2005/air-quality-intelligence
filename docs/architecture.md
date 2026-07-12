# Architecture

```
                    ┌─────────────────────┐
                    │   OpenAQ v3 API      │  (free, CPCB-backed, needs free key)
                    │  ingestion/openaq_   │
                    │      client.py       │
                    └──────────┬───────────┘
                               │ tidy DataFrame
                               │ (city, station, lat, lon, aqi, timestamp)
                               ▼
        ┌──────────────────────────────────────────────┐
        │            data/sample_aqi_data.csv           │  ← offline fallback,
        │   (generated locally, same schema as live)     │    also used to
        └──────────────────────┬─────────────────────────┘   pre-train forecaster
                               │
             ┌─────────────────┼──────────────────┐
             ▼                                     ▼
  ┌────────────────────┐              ┌─────────────────────────┐
  │ forecasting/        │              │ forecasting/             │
  │ aqi_forecast.py      │              │ source_attribution.py    │
  │ RandomForest,         │              │ rule-based time-pattern  │
  │ recursive 24-72h       │              │ classifier                │
  │ forecast + baseline     │              └───────────┬─────────────┘
  │ comparison                │                          │
  └───────────┬────────────────┘                          │
              │                                            │
              └──────────────────┬─────────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  dashboard/app.py         │
                     │  Streamlit: map, forecast, │
                     │  advisory, enforcement     │
                     │  priority queue              │
                     └─────────────────────────┘
```

## Why these choices (for judge Q&A)

- **Government AQI endpoint, not raw CPCB scraping**: the project now uses the
  government data portal's free AQI endpoint directly, which avoids the fragility
  of scraping and keeps the ingestion path simple and reproducible.
- **RandomForest over deep learning**: 30 days of hourly per-station data
  (~720 points/station) is too little for an LSTM/Transformer to
  meaningfully outperform a well-featured tree ensemble, and RF gives free,
  honest `feature_importances_` for explainability during judging.
- **Rule-based source attribution, not a fake ML claim**: real source
  attribution needs land-use polygons, traffic counts, and industrial
  registries with no free, national, machine-readable source in India.
  Claiming an ML classifier here without real features would collapse
  under judge questioning. The time-pattern heuristic is transparent,
  defensible, and correctly upgradeable (see below).

## What's real vs. what's a documented placeholder

| Component | Status |
|---|---|
| Data ingestion (government AQI API) | Real, live API, free key |
| Forecasting model | Real, trained on real/sample history, benchmarked vs baseline |
| Map + dashboard | Real, fully interactive |
| Health advisory text | Real logic, English only |
| Source attribution | Real heuristic, explicitly time-pattern-only |
| Satellite (Sentinel-5P) | **Not implemented** — see upgrade path below |
| Multilingual advisories | **Not implemented** — stub only |

## Recommended next steps (in priority order for remaining hackathon time)

1. **Use the provided government API key** and set `GOVT_AQI_API_KEY` before the
   demo — this switches the app from sample data to live station readings.

2. **Add NASA FIRMS fire/thermal anomaly data** (also free, no key needed
   for basic use: `https://firms.modaps.eosdis.nasa.gov/api/`). This is
   the single highest-leverage improvement: it turns "regional_background"
   attribution from a guess into an evidenced call (crop burning vs.
   dust vs. meteorology), and directly answers the problem statement's
   ask for satellite-informed attribution — without needing full Sentinel-5P
   complexity.

3. **Sentinel-5P via Google Earth Engine** (free tier, needs a free GEE
   signup + `earthengine-api` package). Pulls NO2/aerosol index layers.
   Heavier to set up under time pressure — do this only if steps 1-2 are
   done and you have >4 hours left. A stub function signature is left in
   `ingestion/` for this — `fetch_sentinel5p_layer(bbox, date)` — intentionally
   unimplemented; wire it in only if time allows, and say so plainly on
   the architecture slide rather than faking a call.

4. **OSM land-use polygons** (`overpass-api.de`, free, no key) to replace
   the pure time-pattern heuristic with a spatial join: "this station sits
   inside a residential/traffic-corridor/industrial polygon" — meaningfully
   strengthens source attribution confidence.

5. **Multilingual advisories**: swap the hardcoded English strings in
   `health_advisory()` for calls to a free-tier LLM API (or even a fixed
   translation dictionary for the 5-6 languages CPCB's target cities need)
   — cheap to add and directly matches "Bengaluru in Kannada, Chennai in
   Tamil" from the problem statement.

6. **Multi-city comparative view**: the sidebar already supports switching
   cities; a quick win is a second tab that overlays 2-3 cities' AQI trends
   on one chart for the "Multi-City Comparative Intelligence Dashboard"
   bullet in the problem statement.

## Known limitations to state upfront in your deck (don't let judges find them first)

- Forecast horizon accuracy degrades past ~48h (recursive forecasting
  compounds error) — say this out loud, then show the MAE-vs-baseline
  number as your honest evidence.
- Source attribution is heuristic, not causal — clearly label it as v1
  in the demo.
- Sample data is synthetic when no API key is present — say so if asked,
  don't imply it's real CPCB data unless you've set the key.
