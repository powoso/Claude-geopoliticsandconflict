"""Conflict escalation scoring from ACLED event data.

Computes composite escalation scores using frequency, severity, and
spatial concentration of conflict events over rolling time windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EscalationScore:
    """Multi-dimensional escalation score for a country/region."""

    country: str
    timestamp: datetime
    # Component scores (0 to 1)
    frequency_score: float  # event rate relative to baseline
    severity_score: float  # fatality-weighted severity
    spatial_spread_score: float  # geographic dispersion of events
    actor_diversity_score: float  # number of distinct armed groups
    civilian_targeting_score: float  # proportion targeting civilians
    # Composite
    composite_score: float
    trend: str  # escalating, stable, de_escalating
    components: dict[str, float]


def compute_rolling_event_rate(
    events_df: pd.DataFrame,
    window_days: int = 7,
    baseline_days: int = 90,
) -> pd.DataFrame:
    """Compute rolling event rate and its deviation from baseline.

    Returns daily event counts with z-score against the baseline period.
    """
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    daily = df.groupby("event_date").size().reset_index(name="event_count")
    daily = daily.set_index("event_date").asfreq("D", fill_value=0)

    daily["rolling_mean"] = daily["event_count"].rolling(window=window_days).mean()
    daily["baseline_mean"] = daily["event_count"].rolling(window=baseline_days).mean()
    daily["baseline_std"] = daily["event_count"].rolling(window=baseline_days).std()

    daily["z_score"] = (daily["rolling_mean"] - daily["baseline_mean"]) / daily[
        "baseline_std"
    ].replace(0, 1)

    return daily.reset_index()


def compute_fatality_trend(
    events_df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Compute rolling fatality trends."""
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    daily = df.groupby("event_date")["fatalities"].sum().reset_index()
    daily = daily.set_index("event_date").asfreq("D", fill_value=0)
    daily["rolling_fatalities"] = daily["fatalities"].rolling(window=window_days).sum()
    daily["fatality_trend"] = daily["rolling_fatalities"].pct_change(periods=window_days)
    return daily.reset_index()


def compute_spatial_spread(events_df: pd.DataFrame) -> float:
    """Measure geographic spread of conflict events.

    Uses standard deviation of lat/lon coordinates as a proxy for
    spatial dispersion. Higher spread = more widespread conflict.
    Returns a 0-1 normalized score.
    """
    if events_df.empty or len(events_df) < 2:
        return 0.0

    lat_std = events_df["latitude"].std()
    lon_std = events_df["longitude"].std()

    # Combine into a single spread measure (degrees)
    spread = np.sqrt(lat_std**2 + lon_std**2)

    # Normalize: 0 degrees (concentrated) to ~10 degrees (very spread)
    return float(min(spread / 10.0, 1.0))


def compute_actor_diversity(events_df: pd.DataFrame) -> float:
    """Measure the diversity of armed actors involved.

    More distinct actor groups suggests a more complex, harder-to-resolve conflict.
    Returns 0-1 score based on unique actor count.
    """
    if events_df.empty:
        return 0.0

    actors = set()
    for col in ["actor1", "actor2"]:
        if col in events_df.columns:
            actors.update(events_df[col].dropna().unique())

    # Remove empty strings
    actors.discard("")
    n_actors = len(actors)

    # Normalize: 1 actor = 0, 20+ actors = 1
    return float(min(max(n_actors - 1, 0) / 19.0, 1.0))


def compute_civilian_targeting_ratio(events_df: pd.DataFrame) -> float:
    """Compute the proportion of events that target civilians.

    Higher civilian targeting indicates more severe/desperate conflict dynamics.
    """
    if events_df.empty:
        return 0.0

    total = len(events_df)
    civilian_events = len(
        events_df[events_df["event_type"] == "Violence against civilians"]
    )
    return civilian_events / total if total > 0 else 0.0


def compute_escalation_score(
    events_df: pd.DataFrame,
    country: str,
    short_window: int = 7,
    long_window: int = 90,
    weights: dict[str, float] | None = None,
) -> EscalationScore:
    """Compute composite escalation score for a country.

    Combines multiple conflict indicators into a single 0-1 score
    with trend classification.

    Args:
        events_df: ACLED event DataFrame for the country.
        country: Country name.
        short_window: Days for recent activity window.
        long_window: Days for baseline period.
        weights: Optional custom weights for component scores.
    """
    w = weights or {
        "frequency": 0.25,
        "severity": 0.25,
        "spatial_spread": 0.15,
        "actor_diversity": 0.15,
        "civilian_targeting": 0.20,
    }

    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=short_window)
    baseline_cutoff = now - timedelta(days=long_window)

    df = events_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    recent_df = df[df["event_date"] >= recent_cutoff]
    baseline_df = df[df["event_date"] >= baseline_cutoff]

    # Frequency score
    recent_rate = len(recent_df) / max(short_window, 1)
    baseline_rate = len(baseline_df) / max(long_window, 1)
    freq_ratio = recent_rate / baseline_rate if baseline_rate > 0 else 1.0
    frequency_score = min(freq_ratio / 3.0, 1.0)  # normalize: 3x baseline = 1.0

    # Severity score (fatality-weighted)
    if "weighted_severity" in recent_df.columns and not recent_df.empty:
        severity_score = min(recent_df["weighted_severity"].mean() / 5.0, 1.0)
    elif not recent_df.empty:
        severity_score = min(recent_df["fatalities"].sum() / (short_window * 50), 1.0)
    else:
        severity_score = 0.0

    spatial_score = compute_spatial_spread(recent_df)
    actor_score = compute_actor_diversity(recent_df)
    civilian_score = compute_civilian_targeting_ratio(recent_df)

    composite = (
        w["frequency"] * frequency_score
        + w["severity"] * severity_score
        + w["spatial_spread"] * spatial_score
        + w["actor_diversity"] * actor_score
        + w["civilian_targeting"] * civilian_score
    )

    # Trend detection
    if len(baseline_df) > long_window // 2:
        mid = now - timedelta(days=long_window // 2)
        first_half_rate = len(baseline_df[baseline_df["event_date"] < mid]) / (long_window // 2)
        second_half_rate = len(baseline_df[baseline_df["event_date"] >= mid]) / (long_window // 2)
        if second_half_rate > first_half_rate * 1.3:
            trend = "escalating"
        elif second_half_rate < first_half_rate * 0.7:
            trend = "de_escalating"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    components = {
        "frequency": frequency_score,
        "severity": severity_score,
        "spatial_spread": spatial_score,
        "actor_diversity": actor_score,
        "civilian_targeting": civilian_score,
    }

    return EscalationScore(
        country=country,
        timestamp=now,
        frequency_score=frequency_score,
        severity_score=severity_score,
        spatial_spread_score=spatial_score,
        actor_diversity_score=actor_score,
        civilian_targeting_score=civilian_score,
        composite_score=float(composite),
        trend=trend,
        components=components,
    )
