"""Prediction market contract tracking for geopolitical events.

Aggregates prices from major prediction markets (Polymarket, Metaculus, etc.)
to provide market-implied probabilities that serve as both inputs and
benchmarks for the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketContract:
    """A single prediction market contract."""

    market_id: str
    platform: str  # polymarket, metaculus, manifold, etc.
    question: str
    category: str
    current_price: float  # 0 to 1, representing probability
    volume_24h: float
    total_volume: float
    open_interest: float
    resolution_date: datetime | None
    last_updated: datetime
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """Aggregated market data for a geopolitical topic."""

    topic: str
    contracts: list[MarketContract]
    avg_probability: float
    volume_weighted_probability: float
    market_count: int
    timestamp: datetime


# Known geopolitical prediction market endpoints
POLYMARKET_API = "https://clob.polymarket.com"
METACULUS_API = "https://www.metaculus.com/api2"
MANIFOLD_API = "https://api.manifold.markets/v0"


class PolymarketClient:
    """Client for Polymarket's CLOB API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "GeopoliticalPredictionMarkets/0.1"}
        )

    def search_markets(
        self,
        query: str,
        limit: int = 20,
    ) -> list[MarketContract]:
        """Search Polymarket for geopolitical contracts."""
        try:
            resp = self.session.get(
                f"{POLYMARKET_API}/markets",
                params={"limit": limit, "active": True},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Polymarket API request failed")
            return []

        contracts = []
        query_lower = query.lower()
        for market in data if isinstance(data, list) else data.get("data", []):
            question = market.get("question", "")
            if query_lower not in question.lower():
                continue

            try:
                contracts.append(
                    MarketContract(
                        market_id=str(market.get("condition_id", "")),
                        platform="polymarket",
                        question=question,
                        category=market.get("category", "geopolitics"),
                        current_price=float(market.get("outcomePrices", [0.5])[0]),
                        volume_24h=float(market.get("volume24hr", 0)),
                        total_volume=float(market.get("volumeNum", 0)),
                        open_interest=float(market.get("liquidityNum", 0)),
                        resolution_date=datetime.fromisoformat(market["end_date_iso"])
                        if market.get("end_date_iso")
                        else None,
                        last_updated=datetime.utcnow(),
                        url=f"https://polymarket.com/event/{market.get('slug', '')}",
                    )
                )
            except (ValueError, KeyError):
                continue
        return contracts


class MetaculusClient:
    """Client for Metaculus prediction platform API."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def search_questions(
        self,
        query: str,
        limit: int = 20,
    ) -> list[MarketContract]:
        """Search Metaculus for geopolitical forecast questions."""
        try:
            resp = self.session.get(
                f"{METACULUS_API}/questions/",
                params={
                    "search": query,
                    "limit": limit,
                    "status": "open",
                    "type": "forecast",
                    "order_by": "-activity",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Metaculus API request failed")
            return []

        contracts = []
        for q in data.get("results", []):
            try:
                prediction = q.get("community_prediction", {})
                prob = prediction.get("full", {}).get("q2", 0.5) if prediction else 0.5

                contracts.append(
                    MarketContract(
                        market_id=str(q.get("id", "")),
                        platform="metaculus",
                        question=q.get("title", ""),
                        category="geopolitics",
                        current_price=float(prob),
                        volume_24h=0,
                        total_volume=float(q.get("number_of_predictions", 0)),
                        open_interest=0,
                        resolution_date=datetime.fromisoformat(
                            q["resolve_time"].replace("Z", "+00:00")
                        )
                        if q.get("resolve_time")
                        else None,
                        last_updated=datetime.utcnow(),
                        url=q.get("url", ""),
                    )
                )
            except (ValueError, KeyError):
                continue
        return contracts


class ManifoldClient:
    """Client for Manifold Markets API."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def search_markets(
        self,
        query: str,
        limit: int = 20,
    ) -> list[MarketContract]:
        """Search Manifold Markets for geopolitical contracts."""
        try:
            resp = self.session.get(
                f"{MANIFOLD_API}/search-markets",
                params={"term": query, "limit": limit, "sort": "liquidity"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Manifold Markets API request failed")
            return []

        contracts = []
        for m in data if isinstance(data, list) else []:
            try:
                contracts.append(
                    MarketContract(
                        market_id=m.get("id", ""),
                        platform="manifold",
                        question=m.get("question", ""),
                        category=m.get("groupSlugs", ["geopolitics"])[0]
                        if m.get("groupSlugs")
                        else "geopolitics",
                        current_price=float(m.get("probability", 0.5)),
                        volume_24h=float(m.get("volume24Hours", 0)),
                        total_volume=float(m.get("totalLiquidity", 0)),
                        open_interest=float(m.get("totalLiquidity", 0)),
                        resolution_date=datetime.fromtimestamp(m["closeTime"] / 1000)
                        if m.get("closeTime")
                        else None,
                        last_updated=datetime.utcnow(),
                        url=m.get("url", ""),
                    )
                )
            except (ValueError, KeyError):
                continue
        return contracts


class PredictionMarketAggregator:
    """Aggregate prediction market data across multiple platforms.

    Provides volume-weighted consensus probabilities and detects
    divergences between model predictions and market prices.
    """

    def __init__(self) -> None:
        self.polymarket = PolymarketClient()
        self.metaculus = MetaculusClient()
        self.manifold = ManifoldClient()
        self._cache: dict[str, MarketSnapshot] = {}

    def search_all(self, query: str) -> MarketSnapshot:
        """Search all platforms for a geopolitical topic and aggregate."""
        all_contracts: list[MarketContract] = []
        all_contracts.extend(self.polymarket.search_markets(query))
        all_contracts.extend(self.metaculus.search_questions(query))
        all_contracts.extend(self.manifold.search_markets(query))

        if not all_contracts:
            snapshot = MarketSnapshot(
                topic=query,
                contracts=[],
                avg_probability=0.5,
                volume_weighted_probability=0.5,
                market_count=0,
                timestamp=datetime.utcnow(),
            )
            self._cache[query] = snapshot
            return snapshot

        avg_prob = sum(c.current_price for c in all_contracts) / len(all_contracts)

        total_vol = sum(c.total_volume for c in all_contracts)
        if total_vol > 0:
            vol_weighted = sum(
                c.current_price * c.total_volume for c in all_contracts
            ) / total_vol
        else:
            vol_weighted = avg_prob

        snapshot = MarketSnapshot(
            topic=query,
            contracts=all_contracts,
            avg_probability=avg_prob,
            volume_weighted_probability=vol_weighted,
            market_count=len(all_contracts),
            timestamp=datetime.utcnow(),
        )
        self._cache[query] = snapshot
        return snapshot

    def compare_to_model(
        self,
        query: str,
        model_probability: float,
    ) -> dict[str, Any]:
        """Compare model prediction to market consensus.

        Identifies divergences that may represent either model insight
        or model error.
        """
        snapshot = self._cache.get(query)
        if not snapshot:
            snapshot = self.search_all(query)

        divergence = model_probability - snapshot.volume_weighted_probability

        return {
            "query": query,
            "model_probability": model_probability,
            "market_probability": snapshot.volume_weighted_probability,
            "market_avg_probability": snapshot.avg_probability,
            "divergence": divergence,
            "abs_divergence": abs(divergence),
            "model_more_bullish": divergence > 0,
            "significant_divergence": abs(divergence) > 0.15,
            "market_count": snapshot.market_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
