"""Social media event detection for crisis escalation monitoring.

Monitors Twitter/X for geopolitical crisis keywords and detects
volume/sentiment spikes that may precede or accompany escalation events.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from geopolitical_prediction.config import TwitterConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CrisisTweet:
    """A relevant social media post about a geopolitical event."""

    tweet_id: str
    text: str
    author_id: str
    created_at: datetime
    lang: str
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1


@dataclass
class CrisisSignal:
    """Aggregated crisis signal from social media volume/sentiment analysis."""

    keyword: str
    time_window_minutes: int
    tweet_count: int
    avg_sentiment: float
    volume_z_score: float
    is_spike: bool
    top_tweets: list[CrisisTweet] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Keywords organized by geopolitical scenario
CRISIS_KEYWORDS: dict[str, list[str]] = {
    "military_escalation": [
        "military buildup",
        "troops deployed",
        "airstrikes",
        "missile launch",
        "nuclear threat",
        "mobilization",
        "martial law",
        "invasion",
        "war declaration",
        "no-fly zone",
    ],
    "sanctions": [
        "sanctions imposed",
        "trade embargo",
        "asset freeze",
        "sanctions evasion",
        "SWIFT ban",
        "export controls",
        "sanctions package",
        "economic sanctions",
    ],
    "diplomacy": [
        "peace talks",
        "ceasefire",
        "summit meeting",
        "diplomatic breakthrough",
        "treaty signed",
        "UN resolution",
        "mediation",
        "de-escalation",
    ],
    "humanitarian": [
        "refugee crisis",
        "humanitarian corridor",
        "civilian casualties",
        "displacement",
        "food crisis",
        "aid blocked",
    ],
}


class TwitterCrisisMonitor:
    """Monitor Twitter/X for geopolitical crisis signals.

    Uses the Twitter v2 API to search for crisis-related keywords
    and detect volume/sentiment anomalies.
    """

    def __init__(self, config: TwitterConfig | None = None):
        self.config = config or get_settings().twitter
        self.session = requests.Session()
        if self.config.bearer_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.config.bearer_token}"}
            )
        self._volume_history: dict[str, list[tuple[datetime, int]]] = defaultdict(list)

    def search_recent(
        self,
        query: str,
        max_results: int = 100,
        start_time: datetime | None = None,
    ) -> list[CrisisTweet]:
        """Search recent tweets matching a query.

        Uses Twitter v2 recent search (last 7 days).
        """
        if not self.config.bearer_token:
            logger.warning("Twitter bearer token not configured")
            return []

        params: dict[str, Any] = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,lang,author_id",
            "sort_order": "relevancy",
        }
        if start_time:
            params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = self.session.get(self.config.search_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Twitter search failed for query=%s", query)
            return []

        tweets = []
        for item in data.get("data", []):
            metrics = item.get("public_metrics", {})
            try:
                tweets.append(
                    CrisisTweet(
                        tweet_id=item["id"],
                        text=item.get("text", ""),
                        author_id=item.get("author_id", ""),
                        created_at=datetime.strptime(
                            item.get("created_at", ""), "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        lang=item.get("lang", "en"),
                        retweet_count=metrics.get("retweet_count", 0),
                        like_count=metrics.get("like_count", 0),
                        reply_count=metrics.get("reply_count", 0),
                        quote_count=metrics.get("quote_count", 0),
                        matched_keywords=[query],
                    )
                )
            except (ValueError, KeyError):
                continue
        return tweets

    def compute_volume_signal(
        self,
        keyword: str,
        window_minutes: int = 60,
        history_hours: int = 24,
    ) -> CrisisSignal:
        """Detect volume anomalies for a keyword.

        Compares recent tweet volume to historical baseline using z-score.
        A spike (z > 2) suggests a breaking event.
        """
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=window_minutes)

        tweets = self.search_recent(keyword, max_results=100, start_time=start_time)
        current_count = len(tweets)

        # Update volume history
        self._volume_history[keyword].append((now, current_count))
        # Trim old history
        cutoff = now - timedelta(hours=history_hours)
        self._volume_history[keyword] = [
            (t, c) for t, c in self._volume_history[keyword] if t >= cutoff
        ]

        # Compute z-score against history
        counts = [c for _, c in self._volume_history[keyword]]
        if len(counts) >= 3:
            mean_count = sum(counts) / len(counts)
            std_count = (sum((c - mean_count) ** 2 for c in counts) / len(counts)) ** 0.5
            z_score = (current_count - mean_count) / std_count if std_count > 0 else 0
        else:
            z_score = 0.0

        # Basic sentiment from tweet text
        avg_sentiment = 0.0
        if tweets:
            try:
                from textblob import TextBlob

                sentiments = [TextBlob(t.text).sentiment.polarity for t in tweets]
                avg_sentiment = sum(sentiments) / len(sentiments)
                for tweet, sent in zip(tweets, sentiments):
                    tweet.sentiment = sent
            except ImportError:
                pass

        is_spike = z_score > 2.0

        return CrisisSignal(
            keyword=keyword,
            time_window_minutes=window_minutes,
            tweet_count=current_count,
            avg_sentiment=avg_sentiment,
            volume_z_score=z_score,
            is_spike=is_spike,
            top_tweets=sorted(tweets, key=lambda t: t.retweet_count, reverse=True)[:5],
            timestamp=now,
        )

    def scan_all_keywords(
        self, scenario: str | None = None
    ) -> list[CrisisSignal]:
        """Scan all crisis keywords and return signals.

        Args:
            scenario: Optional filter to scan only keywords for a specific
                      scenario (e.g., "military_escalation", "sanctions").
        """
        if scenario:
            keywords = CRISIS_KEYWORDS.get(scenario, [])
        else:
            keywords = [kw for group in CRISIS_KEYWORDS.values() for kw in group]

        signals = []
        for keyword in keywords:
            signal = self.compute_volume_signal(keyword)
            signals.append(signal)
        return signals

    def get_active_spikes(self, scenario: str | None = None) -> list[CrisisSignal]:
        """Return only keywords currently experiencing volume spikes."""
        signals = self.scan_all_keywords(scenario)
        return [s for s in signals if s.is_spike]
