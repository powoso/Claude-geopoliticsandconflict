"""Bilateral relationship sentiment tracking from GDELT tone analysis.

Transforms raw GDELT tone time-series into features for the
probability engine: trend direction, volatility, and anomaly detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentFeatures:
    """Extracted sentiment features for a country pair."""

    country_a: str
    country_b: str
    timestamp: datetime
    # Current state
    current_tone: float  # latest average tone
    tone_7d_avg: float  # 7-day moving average
    tone_30d_avg: float  # 30-day moving average
    # Dynamics
    tone_trend: float  # slope of linear fit over 30 days
    tone_volatility: float  # standard deviation of daily tone
    tone_z_score: float  # current tone vs 90-day baseline
    # Volume
    article_count_7d: int
    article_volume_trend: float  # change in media attention
    # Derived
    sentiment_category: str  # hostile, negative, neutral, positive, cooperative
    is_deteriorating: bool
    is_crisis_level: bool


def extract_sentiment_features(
    tone_df: pd.DataFrame,
    country_a: str,
    country_b: str,
) -> SentimentFeatures | None:
    """Extract sentiment features from a GDELT bilateral tone DataFrame.

    Args:
        tone_df: DataFrame with columns [date, avg_tone, article_count].
        country_a: First country in the pair.
        country_b: Second country in the pair.
    """
    if tone_df.empty or len(tone_df) < 7:
        return None

    df = tone_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    tones = df["avg_tone"].values
    current_tone = float(tones[-1])

    # Moving averages
    tone_7d = float(np.mean(tones[-7:])) if len(tones) >= 7 else current_tone
    tone_30d = float(np.mean(tones[-30:])) if len(tones) >= 30 else float(np.mean(tones))

    # Trend: linear regression slope over last 30 days
    window = min(30, len(tones))
    x = np.arange(window)
    y = tones[-window:]
    if len(x) > 1:
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    # Volatility
    volatility = float(np.std(tones[-30:])) if len(tones) >= 30 else float(np.std(tones))

    # Z-score against 90-day baseline
    baseline = tones[-90:] if len(tones) >= 90 else tones
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    z_score = (current_tone - baseline_mean) / baseline_std if baseline_std > 0 else 0.0

    # Article volume
    if "article_count" in df.columns:
        article_7d = int(df["article_count"].tail(7).sum())
        vol_recent = df["article_count"].tail(7).mean()
        vol_baseline = df["article_count"].tail(30).mean()
        volume_trend = float(
            (vol_recent - vol_baseline) / vol_baseline if vol_baseline > 0 else 0
        )
    else:
        article_7d = 0
        volume_trend = 0.0

    # Categorize sentiment
    if current_tone < -5:
        category = "hostile"
    elif current_tone < -2:
        category = "negative"
    elif current_tone < 2:
        category = "neutral"
    elif current_tone < 5:
        category = "positive"
    else:
        category = "cooperative"

    is_deteriorating = slope < -0.1 and z_score < -1.0
    is_crisis = current_tone < -5 and z_score < -2.0

    return SentimentFeatures(
        country_a=country_a,
        country_b=country_b,
        timestamp=datetime.utcnow(),
        current_tone=current_tone,
        tone_7d_avg=tone_7d,
        tone_30d_avg=tone_30d,
        tone_trend=slope,
        tone_volatility=volatility,
        tone_z_score=z_score,
        article_count_7d=article_7d,
        article_volume_trend=volume_trend,
        sentiment_category=category,
        is_deteriorating=is_deteriorating,
        is_crisis_level=is_crisis,
    )


def compute_sentiment_trajectory(
    tone_df: pd.DataFrame,
    window_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """Compute multi-scale sentiment trajectory features.

    Returns a DataFrame with moving averages, volatility bands,
    and momentum indicators at multiple time scales.
    """
    if tone_df.empty:
        return pd.DataFrame()

    windows = window_sizes or [3, 7, 14, 30]
    df = tone_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    for w in windows:
        df[f"tone_ma_{w}d"] = df["avg_tone"].rolling(window=w, min_periods=1).mean()
        df[f"tone_std_{w}d"] = df["avg_tone"].rolling(window=w, min_periods=1).std()

    # Momentum: difference between short and long MA
    if len(windows) >= 2:
        short_w = windows[0]
        long_w = windows[-1]
        df["momentum"] = df[f"tone_ma_{short_w}d"] - df[f"tone_ma_{long_w}d"]
        df["momentum_direction"] = df["momentum"].apply(
            lambda x: "improving" if x > 0.5 else "deteriorating" if x < -0.5 else "stable"
        )

    return df.reset_index()


def bilateral_sentiment_divergence(
    pair_a_df: pd.DataFrame,
    pair_b_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare sentiment trajectories of two country pairs.

    Useful for detecting asymmetric escalation: if A-B sentiment
    deteriorates while A-C remains stable, it localizes the tension.
    """
    if pair_a_df.empty or pair_b_df.empty:
        return pd.DataFrame()

    a = pair_a_df[["date", "avg_tone"]].rename(columns={"avg_tone": "tone_pair_a"})
    b = pair_b_df[["date", "avg_tone"]].rename(columns={"avg_tone": "tone_pair_b"})

    merged = a.merge(b, on="date", how="inner")
    merged["divergence"] = merged["tone_pair_a"] - merged["tone_pair_b"]
    merged["abs_divergence"] = merged["divergence"].abs()

    return merged
