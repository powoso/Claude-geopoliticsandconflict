# GeoPred — Geopolitical Prediction Markets

A Python system for probabilistic forecasting of geopolitical events: invasions, sanctions, trade deals, ceasefires, and more.

Combines data from GDELT, ACLED, SIPRI, commodity markets, satellite imagery, ship tracking, social media, and prediction markets into an ensemble probability engine with a polished Streamlit dashboard.

## Quick Start (macOS)

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Homebrew** (for installing system dependencies)

### 1. Clone the repository

```bash
git clone https://github.com/powoso/Claude-geopoliticsandconflict.git
cd Claude-geopoliticsandconflict
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you run into issues with `geopandas` on Apple Silicon:

```bash
brew install gdal geos proj
pip install --no-binary :all: fiona
pip install -r requirements.txt
```

### 4. Configure API keys (optional)

Copy the example config and fill in any API keys you have:

```bash
cp .env.example .env
```

Keys you can configure:

| Service | Key | Required? |
|---------|-----|-----------|
| ACLED | `ACLED_API_KEY`, `ACLED_EMAIL` | For conflict event data |
| Sentinel Hub | `SENTINEL_HUB_CLIENT_ID`, `SENTINEL_HUB_CLIENT_SECRET` | For satellite imagery |
| MarineTraffic | `MARINE_TRAFFIC_API_KEY` | For ship tracking |
| Twitter/X | `TWITTER_BEARER_TOKEN` | For social media monitoring |

**GDELT requires no API key** — the v2 API is free and open.

Prediction market APIs (Polymarket, Metaculus, Manifold) are also free and keyless.

### 5. Initialize the database

```bash
python -m geopolitical_prediction.cli init-db
```

### 6. Run the dashboard

```bash
streamlit run geopolitical_prediction/dashboard/app.py
```

The dashboard opens at `http://localhost:8501` with a dark-themed UI featuring:
- Global conflict heatmap
- Active market forecasts with probability gauges
- Escalation monitor with country drill-down
- Bilateral sentiment tracker (GDELT tone time-series)
- Alliance network explorer and intervention simulator
- Commodity stress monitor
- Historical analog matching
- Base rate reference with regime-type modifiers
- Calibration and accuracy tracking
- Live alert feed

### 7. Run the CLI

```bash
# List all prediction market questions
python -m geopolitical_prediction.cli list-questions

# Generate probability forecasts
python -m geopolitical_prediction.cli forecast

# Show historical base rates
python -m geopolitical_prediction.cli base-rates

# Update data from GDELT (no API key needed)
python -m geopolitical_prediction.cli update-data --source gdelt --days 30

# Update data from ACLED (requires API key)
python -m geopolitical_prediction.cli update-data --source acled --days 90
```

## Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 60 tests pass without any API keys or network access — they use mocked data.

## Architecture

```
geopolitical_prediction/
├── config.py                     # Pydantic settings for all APIs and thresholds
├── cli.py                        # CLI: update-data, forecast, dashboard, etc.
├── data/                         # 9 data collectors
│   ├── gdelt.py                  # GDELT v2 DOC/GEO/GKG APIs + sentiment pipeline
│   ├── acled.py                  # ACLED conflict events + escalation detection
│   ├── commodities.py            # Yahoo Finance commodity prices
│   ├── un_votes.py               # UN GA voting alignment + coalition detection
│   ├── sipri.py                  # Arms transfer analysis + buildup detection
│   ├── satellite.py              # Sentinel Hub NDVI change detection
│   ├── ship_tracking.py          # AIS vessel monitoring at chokepoints
│   ├── social_media.py           # Twitter/X crisis keyword spike detection
│   └── prediction_markets.py     # Polymarket/Metaculus/Manifold aggregation
├── features/                     # 6 feature engineering modules
│   ├── conflict_scoring.py       # Multi-dimensional escalation scoring
│   ├── sentiment.py              # Bilateral tone trajectory + momentum
│   ├── base_rates.py             # Historical rates with regime-type modifiers
│   ├── economic.py               # Trade interdependence as escalation constraint
│   ├── political.py              # Diversionary war theory indicators
│   └── alliance.py               # NetworkX alliance graph + intervention probability
├── models/                       # 3 probability models
│   ├── escalation_ladder.py      # 6-rung Bayesian HMM with evidence updates
│   ├── analog_matching.py        # 12 coded historical crises for comparison
│   └── ensemble.py               # Weighted signal combination + calibration
├── storage/database.py           # SQLAlchemy ORM (SQLite/PostgreSQL)
├── markets/questions.py          # 11 predefined geopolitical questions
└── dashboard/
    ├── alerts.py                 # Rule-based alert engine
    └── app.py                    # Streamlit dashboard (9 pages)
```

## Data Sources

| Source | What it provides | API Key? |
|--------|-----------------|----------|
| **GDELT** | Global news sentiment, bilateral tone tracking, 15-min updates | Free |
| **ACLED** | Armed conflict events with geolocation, fatalities, actors | Free (register) |
| **Yahoo Finance** | Commodity prices (oil, gold, wheat, etc.) | Free |
| **Polymarket** | Prediction market contract prices | Free |
| **Metaculus** | Community forecasting probabilities | Free |
| **Manifold** | Prediction market prices and liquidity | Free |
| **SIPRI** | Arms transfer data (CSV export) | Free |
| **UN Dataverse** | General Assembly voting records (CSV) | Free |
| **Sentinel Hub** | Satellite imagery for change detection | Free tier |
| **MarineTraffic** | AIS vessel tracking at chokepoints | Paid |
| **Twitter/X** | Crisis keyword monitoring | Paid |

## Models

### Escalation Ladder (Bayesian HMM)
Six discrete rungs from Normal Relations to Full-Scale Conflict. Evidence from ACLED, GDELT, and economic indicators updates transition probabilities via likelihood ratios. Forward simulation via matrix exponentiation produces outcome distributions.

### Historical Analog Matching
12 coded crises (Cuban Missile Crisis through Russia-Ukraine 2022) scored on 8 dimensions. Similarity-weighted outcome probabilities provide empirically-grounded forecasts.

### Ensemble Engine
Combines base rates, escalation model, analog matcher, sentiment signals, conflict scores, economic constraints, and market prices. Tracks calibration via Brier score.

## Active Market Questions

The system ships with 11 predefined questions spanning:
- **Invasion:** China-Taiwan, Russia-NATO, North Korea-South Korea, India-Pakistan
- **Sanctions:** US-China tech, EU-Russia, Iran nuclear
- **Trade deals:** US-UK, India-RCEP
- **Ceasefires:** Ukraine, Sudan

Create custom questions via `markets.questions.create_custom_question()`.

## License

MIT
