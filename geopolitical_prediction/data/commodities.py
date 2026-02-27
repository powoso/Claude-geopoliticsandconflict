"""Commodity price tracking as geopolitical stress proxies.

Oil, wheat, and metals prices react to geopolitical instability.
Sudden spikes often precede or accompany conflict escalation and
sanctions regimes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Free commodity data sources
YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Key commodities as geopolitical indicators
COMMODITY_SYMBOLS: dict[str, dict[str, str]] = {
    "crude_oil_wti": {"symbol": "CL=F", "name": "WTI Crude Oil"},
    "crude_oil_brent": {"symbol": "BZ=F", "name": "Brent Crude Oil"},
    "natural_gas": {"symbol": "NG=F", "name": "Natural Gas"},
    "gold": {"symbol": "GC=F", "name": "Gold"},
    "wheat": {"symbol": "ZW=F", "name": "Wheat"},
    "corn": {"symbol": "ZC=F", "name": "Corn"},
    "copper": {"symbol": "HG=F", "name": "Copper"},
    "palladium": {"symbol": "PA=F", "name": "Palladium"},
    "uranium": {"symbol": "UX=F", "name": "Uranium"},
}

# Which commodities are proxies for which geopolitical scenarios
SCENARIO_COMMODITIES: dict[str, list[str]] = {
    "middle_east_conflict": ["crude_oil_wti", "crude_oil_brent", "gold", "natural_gas"],
    "russia_conflict": ["crude_oil_brent", "natural_gas", "wheat", "palladium"],
    "china_taiwan": ["copper", "gold", "crude_oil_brent"],
    "food_crisis": ["wheat", "corn"],
    "general_risk": ["gold", "crude_oil_wti"],
}


class CommodityClient:
    """Fetch commodity price data from Yahoo Finance (free, no API key)."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 GeopoliticalPredictionMarkets/0.1"}
        )

    def fetch_price_history(
        self,
        commodity_key: str,
        days_back: int = 180,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical prices for a commodity.

        Args:
            commodity_key: Key from COMMODITY_SYMBOLS dict.
            days_back: Number of days of history.
            interval: 1d, 1wk, or 1mo.
        """
        info = COMMODITY_SYMBOLS.get(commodity_key)
        if not info:
            logger.error("Unknown commodity key: %s", commodity_key)
            return pd.DataFrame()

        end_ts = int(datetime.utcnow().timestamp())
        start_ts = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())

        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "events": "history",
        }

        try:
            resp = self.session.get(
                YAHOO_FINANCE_URL.format(symbol=info["symbol"]),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Failed to fetch commodity data for %s", commodity_key)
            return pd.DataFrame()

        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()

        timestamps = result[0].get("timestamp", [])
        quotes = result[0].get("indicators", {}).get("quote", [{}])[0]

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s"),
                "open": quotes.get("open", []),
                "high": quotes.get("high", []),
                "low": quotes.get("low", []),
                "close": quotes.get("close", []),
                "volume": quotes.get("volume", []),
            }
        )
        df["commodity"] = commodity_key
        df["name"] = info["name"]
        df = df.dropna(subset=["close"])
        return df.reset_index(drop=True)

    def fetch_scenario_commodities(
        self,
        scenario: str,
        days_back: int = 180,
    ) -> dict[str, pd.DataFrame]:
        """Fetch price data for all commodities relevant to a geopolitical scenario."""
        commodities = SCENARIO_COMMODITIES.get(scenario, [])
        results: dict[str, pd.DataFrame] = {}
        for key in commodities:
            df = self.fetch_price_history(key, days_back)
            if not df.empty:
                results[key] = df
        return results


class CommodityStressDetector:
    """Detect commodity price anomalies as geopolitical stress signals."""

    def __init__(self) -> None:
        self.client = CommodityClient()
        self._cache: dict[str, pd.DataFrame] = {}

    def update(self, commodities: list[str] | None = None, days_back: int = 180) -> None:
        """Refresh price data for specified commodities."""
        keys = commodities or list(COMMODITY_SYMBOLS.keys())
        for key in keys:
            df = self.client.fetch_price_history(key, days_back)
            if not df.empty:
                self._cache[key] = df

    def compute_stress_signals(
        self,
        commodity_key: str,
        short_window: int = 5,
        long_window: int = 60,
    ) -> dict[str, Any] | None:
        """Compute price stress signals for a commodity.

        Returns z-score of recent price move vs historical volatility,
        plus trend direction and magnitude.
        """
        df = self._cache.get(commodity_key)
        if df is None or len(df) < long_window:
            return None

        prices = df["close"].values
        recent_return = (prices[-1] - prices[-short_window]) / prices[-short_window]
        long_returns = pd.Series(prices).pct_change(short_window).dropna()
        mean_return = long_returns.tail(long_window).mean()
        std_return = long_returns.tail(long_window).std()

        z_score = (recent_return - mean_return) / std_return if std_return > 0 else 0

        return {
            "commodity": commodity_key,
            "current_price": float(prices[-1]),
            "short_window_return": float(recent_return),
            "z_score": float(z_score),
            "is_stressed": abs(z_score) > 2.0,
            "direction": "spike" if z_score > 0 else "crash",
            "volatility": float(std_return),
        }

    def scenario_stress_index(self, scenario: str) -> dict[str, Any]:
        """Compute composite stress index for a geopolitical scenario.

        Averages the z-scores of all relevant commodities.
        """
        commodities = SCENARIO_COMMODITIES.get(scenario, [])
        signals = []
        for key in commodities:
            sig = self.compute_stress_signals(key)
            if sig:
                signals.append(sig)

        if not signals:
            return {"scenario": scenario, "stress_index": 0.0, "signals": []}

        avg_z = sum(s["z_score"] for s in signals) / len(signals)
        return {
            "scenario": scenario,
            "stress_index": float(avg_z),
            "is_stressed": abs(avg_z) > 1.5,
            "signals": signals,
        }
