"""Economic interdependence metrics as conflict constraint indicators.

High bilateral trade volumes, mutual debt holdings, and supply chain
dependencies constrain escalation. These features quantify how much
countries have to lose from conflict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EconomicInterdependence:
    """Economic interdependence metrics between two countries."""

    country_a: str
    country_b: str
    # Trade metrics
    bilateral_trade_volume: float  # USD
    trade_as_pct_gdp_a: float  # trade with B as % of A's GDP
    trade_as_pct_gdp_b: float
    trade_balance: float  # positive = A exports more to B
    # Dependency metrics
    import_dependency_a: float  # A's imports from B / A's total imports
    import_dependency_b: float
    critical_supply_chains: list[str]  # sectors where dependency is high
    # Financial links
    debt_holdings_a_of_b: float  # A holds B's sovereign debt (USD)
    debt_holdings_b_of_a: float
    fdi_a_in_b: float  # A's foreign direct investment in B
    fdi_b_in_a: float
    # Composite
    interdependence_score: float  # 0 to 1, higher = more interdependent
    escalation_constraint: float  # 0 to 1, how much economics constrains escalation


# Key bilateral economic relationships with approximate 2023 data
BILATERAL_TRADE_DATA: dict[str, dict[str, Any]] = {
    "US_China": {
        "country_a": "United States",
        "country_b": "China",
        "bilateral_trade_volume": 758_000_000_000,
        "trade_as_pct_gdp_a": 2.8,
        "trade_as_pct_gdp_b": 4.2,
        "critical_supply_chains": [
            "semiconductors",
            "rare_earths",
            "pharmaceuticals",
            "electronics",
        ],
        "debt_holdings_a_of_b": 0,
        "debt_holdings_b_of_a": 816_000_000_000,
    },
    "Russia_EU": {
        "country_a": "Russia",
        "country_b": "EU",
        "bilateral_trade_volume": 258_000_000_000,
        "trade_as_pct_gdp_a": 14.0,
        "trade_as_pct_gdp_b": 1.5,
        "critical_supply_chains": ["natural_gas", "oil", "metals", "wheat"],
        "debt_holdings_a_of_b": 0,
        "debt_holdings_b_of_a": 0,
    },
    "China_Taiwan": {
        "country_a": "China",
        "country_b": "Taiwan",
        "bilateral_trade_volume": 320_000_000_000,
        "trade_as_pct_gdp_a": 1.8,
        "trade_as_pct_gdp_b": 42.0,
        "critical_supply_chains": [
            "semiconductors",
            "electronics",
            "machinery",
        ],
        "debt_holdings_a_of_b": 0,
        "debt_holdings_b_of_a": 0,
    },
    "India_Pakistan": {
        "country_a": "India",
        "country_b": "Pakistan",
        "bilateral_trade_volume": 2_500_000_000,
        "trade_as_pct_gdp_a": 0.07,
        "trade_as_pct_gdp_b": 0.7,
        "critical_supply_chains": [],
        "debt_holdings_a_of_b": 0,
        "debt_holdings_b_of_a": 0,
    },
}


def compute_interdependence_score(metrics: dict[str, Any]) -> float:
    """Compute composite economic interdependence score.

    Combines trade dependency, financial linkages, and supply chain
    criticality into a 0-1 score.
    """
    trade_score = min(
        (metrics.get("trade_as_pct_gdp_a", 0) + metrics.get("trade_as_pct_gdp_b", 0))
        / 30.0,
        1.0,
    )

    supply_chain_score = min(len(metrics.get("critical_supply_chains", [])) / 5.0, 1.0)

    debt_score = min(
        (metrics.get("debt_holdings_a_of_b", 0) + metrics.get("debt_holdings_b_of_a", 0))
        / 1_000_000_000_000,
        1.0,
    )

    return 0.4 * trade_score + 0.35 * supply_chain_score + 0.25 * debt_score


def compute_escalation_constraint(interdependence_score: float) -> float:
    """Convert interdependence score to escalation constraint.

    Higher interdependence means more economic cost to escalation,
    reducing the probability of military action.

    Uses a logistic function to model diminishing returns:
    very high interdependence is a strong constraint, but doesn't
    make conflict impossible (see WWI).
    """
    import numpy as np

    # Logistic curve centered at 0.5 interdependence
    # Max constraint = 0.7 (economics alone can't prevent war)
    return float(0.7 / (1 + np.exp(-8 * (interdependence_score - 0.4))))


class EconomicAnalyzer:
    """Analyze economic interdependence between countries."""

    def __init__(self) -> None:
        self._trade_data = dict(BILATERAL_TRADE_DATA)

    def get_bilateral_metrics(
        self,
        country_a: str,
        country_b: str,
    ) -> EconomicInterdependence | None:
        """Get or compute economic interdependence metrics for a country pair."""
        # Check both orderings
        key_ab = f"{country_a}_{country_b}".replace(" ", "_")
        key_ba = f"{country_b}_{country_a}".replace(" ", "_")

        data = self._trade_data.get(key_ab) or self._trade_data.get(key_ba)
        if not data:
            return None

        score = compute_interdependence_score(data)
        constraint = compute_escalation_constraint(score)

        return EconomicInterdependence(
            country_a=data.get("country_a", country_a),
            country_b=data.get("country_b", country_b),
            bilateral_trade_volume=data.get("bilateral_trade_volume", 0),
            trade_as_pct_gdp_a=data.get("trade_as_pct_gdp_a", 0),
            trade_as_pct_gdp_b=data.get("trade_as_pct_gdp_b", 0),
            trade_balance=data.get("trade_balance", 0),
            import_dependency_a=data.get("import_dependency_a", 0),
            import_dependency_b=data.get("import_dependency_b", 0),
            critical_supply_chains=data.get("critical_supply_chains", []),
            debt_holdings_a_of_b=data.get("debt_holdings_a_of_b", 0),
            debt_holdings_b_of_a=data.get("debt_holdings_b_of_a", 0),
            fdi_a_in_b=data.get("fdi_a_in_b", 0),
            fdi_b_in_a=data.get("fdi_b_in_a", 0),
            interdependence_score=score,
            escalation_constraint=constraint,
        )

    def load_trade_data(self, csv_path: str) -> None:
        """Load bilateral trade data from CSV.

        Expected columns: country_a, country_b, trade_volume, year
        """
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                key = f"{row['country_a']}_{row['country_b']}".replace(" ", "_")
                self._trade_data[key] = row.to_dict()
            logger.info("Loaded %d bilateral trade records", len(df))
        except Exception:
            logger.exception("Failed to load trade data from %s", csv_path)

    def rank_escalation_constraints(self) -> list[dict[str, Any]]:
        """Rank all known bilateral relationships by escalation constraint.

        Higher constraint = conflict is more economically costly.
        """
        results = []
        for key, data in self._trade_data.items():
            score = compute_interdependence_score(data)
            constraint = compute_escalation_constraint(score)
            results.append(
                {
                    "pair": key,
                    "interdependence_score": score,
                    "escalation_constraint": constraint,
                    "trade_volume": data.get("bilateral_trade_volume", 0),
                }
            )
        return sorted(results, key=lambda x: x["escalation_constraint"], reverse=True)
