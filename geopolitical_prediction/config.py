"""Central configuration for all data sources, thresholds, and model parameters."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ACLEDConfig(BaseSettings):
    api_key: str = Field(default="", alias="ACLED_API_KEY")
    email: str = Field(default="", alias="ACLED_EMAIL")
    base_url: str = "https://api.acleddata.com/acled/read"
    page_size: int = 5000

    model_config = {"env_prefix": "ACLED_", "extra": "ignore"}


class GDELTConfig(BaseSettings):
    base_url: str = "https://api.gdeltproject.org/api/v2"
    doc_api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    geo_api_url: str = "https://api.gdeltproject.org/api/v2/geo/geo"
    tv_api_url: str = "https://api.gdeltproject.org/api/v2/tv/tv"
    gkg_url: str = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    request_delay: float = 1.0  # seconds between requests to respect rate limits

    model_config = {"env_prefix": "GDELT_", "extra": "ignore"}


class SentinelConfig(BaseSettings):
    client_id: str = Field(default="", alias="SENTINEL_HUB_CLIENT_ID")
    client_secret: str = Field(default="", alias="SENTINEL_HUB_CLIENT_SECRET")
    oauth_url: str = "https://services.sentinel-hub.com/oauth/token"
    process_url: str = "https://services.sentinel-hub.com/api/v1/process"

    model_config = {"env_prefix": "SENTINEL_", "extra": "ignore"}


class MarineTrafficConfig(BaseSettings):
    api_key: str = Field(default="", alias="MARINE_TRAFFIC_API_KEY")
    base_url: str = "https://services.marinetraffic.com/api"

    model_config = {"env_prefix": "MARINE_TRAFFIC_", "extra": "ignore"}


class TwitterConfig(BaseSettings):
    bearer_token: str = Field(default="", alias="TWITTER_BEARER_TOKEN")
    stream_url: str = "https://api.twitter.com/2/tweets/search/stream"
    search_url: str = "https://api.twitter.com/2/tweets/search/recent"

    model_config = {"env_prefix": "TWITTER_", "extra": "ignore"}


class DatabaseConfig(BaseSettings):
    url: str = Field(
        default="sqlite:///geopolitical_predictions.db", alias="DATABASE_URL"
    )

    model_config = {"env_prefix": "DATABASE_", "extra": "ignore"}


class AlertConfig(BaseSettings):
    escalation_threshold: float = Field(
        default=0.7, alias="ALERT_ESCALATION_THRESHOLD"
    )
    email: str = Field(default="", alias="ALERT_EMAIL")

    model_config = {"env_prefix": "ALERT_", "extra": "ignore"}


class ModelConfig(BaseSettings):
    """Probability engine hyperparameters."""

    # Rolling window sizes for feature computation
    short_window_days: int = 7
    medium_window_days: int = 30
    long_window_days: int = 90

    # Escalation ladder thresholds
    escalation_low: float = 0.2
    escalation_medium: float = 0.5
    escalation_high: float = 0.75
    escalation_critical: float = 0.9

    # Analog matching
    analog_top_k: int = 10
    analog_min_similarity: float = 0.3

    # Ensemble weights (GDELT sentiment, ACLED events, economic indicators)
    ensemble_weight_sentiment: float = 0.3
    ensemble_weight_conflict: float = 0.4
    ensemble_weight_economic: float = 0.3

    model_config = {"env_prefix": "MODEL_", "extra": "ignore"}


class Settings(BaseSettings):
    """Top-level settings aggregating all sub-configs."""

    acled: ACLEDConfig = Field(default_factory=ACLEDConfig)
    gdelt: GDELTConfig = Field(default_factory=GDELTConfig)
    sentinel: SentinelConfig = Field(default_factory=SentinelConfig)
    marine_traffic: MarineTrafficConfig = Field(default_factory=MarineTrafficConfig)
    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    data_cache_dir: Path = Path("data_cache")

    model_config = {"env_file": ".env", "extra": "ignore"}

    def ensure_cache_dir(self) -> Path:
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.data_cache_dir


def get_settings() -> Settings:
    return Settings()
