"""Historical base rate computation for geopolitical events.

Answers questions like:
  - How often do military buildups lead to invasion?
  - How often do sanctions threats materialize?
  - What fraction of ceasefire agreements hold for 1 year?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BaseRateEstimate:
    """A historical base rate for a type of geopolitical event."""

    event_type: str
    description: str
    base_rate: float  # probability (0 to 1)
    sample_size: int
    time_period: str
    source: str
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    conditions: dict[str, Any] = field(default_factory=dict)


# Compiled from academic literature and historical data
# These serve as priors for the Bayesian escalation model
HISTORICAL_BASE_RATES: dict[str, BaseRateEstimate] = {
    "military_buildup_to_invasion": BaseRateEstimate(
        event_type="military_buildup_to_invasion",
        description="Probability that a significant military buildup leads to cross-border invasion",
        base_rate=0.15,
        sample_size=120,
        time_period="1945-2023",
        source="COW MID dataset + manual coding",
        confidence_interval=(0.10, 0.22),
        conditions={"requires": "significant_troop_buildup", "within_months": 12},
    ),
    "sanctions_threat_to_imposition": BaseRateEstimate(
        event_type="sanctions_threat_to_imposition",
        description="Probability that a public sanctions threat leads to actual sanctions",
        base_rate=0.45,
        sample_size=200,
        time_period="1990-2023",
        source="TIES sanctions dataset",
        confidence_interval=(0.38, 0.52),
        conditions={"requires": "public_threat_by_official"},
    ),
    "ceasefire_holds_1yr": BaseRateEstimate(
        event_type="ceasefire_holds_1yr",
        description="Probability that a ceasefire agreement holds for at least 1 year",
        base_rate=0.40,
        sample_size=300,
        time_period="1946-2023",
        source="UCDP ceasefire dataset",
        confidence_interval=(0.35, 0.46),
    ),
    "interstate_crisis_to_war": BaseRateEstimate(
        event_type="interstate_crisis_to_war",
        description="Probability that an interstate crisis escalates to full war (1000+ battle deaths)",
        base_rate=0.05,
        sample_size=450,
        time_period="1918-2023",
        source="ICB crisis dataset",
        confidence_interval=(0.03, 0.08),
    ),
    "trade_deal_completion": BaseRateEstimate(
        event_type="trade_deal_completion",
        description="Probability that announced trade negotiations conclude in a signed deal",
        base_rate=0.35,
        sample_size=150,
        time_period="1980-2023",
        source="DESTA trade agreements database",
        confidence_interval=(0.28, 0.42),
    ),
    "coup_attempt_success": BaseRateEstimate(
        event_type="coup_attempt_success",
        description="Probability that a coup attempt succeeds",
        base_rate=0.50,
        sample_size=475,
        time_period="1950-2023",
        source="Powell & Thyne coup dataset",
        confidence_interval=(0.45, 0.55),
    ),
    "nuclear_threat_to_use": BaseRateEstimate(
        event_type="nuclear_threat_to_use",
        description="Probability that an explicit nuclear threat leads to nuclear weapon use",
        base_rate=0.0,
        sample_size=25,
        time_period="1945-2023",
        source="Nuclear threats database",
        confidence_interval=(0.0, 0.05),
    ),
    "alliance_invocation": BaseRateEstimate(
        event_type="alliance_invocation",
        description="Probability that a mutual defense treaty is invoked when ally is attacked",
        base_rate=0.70,
        sample_size=80,
        time_period="1945-2023",
        source="ATOP alliance treaty dataset",
        confidence_interval=(0.60, 0.80),
    ),
    "un_resolution_compliance": BaseRateEstimate(
        event_type="un_resolution_compliance",
        description="Probability that a UNSC resolution achieves stated objectives",
        base_rate=0.30,
        sample_size=250,
        time_period="1945-2023",
        source="UN resolution outcomes analysis",
        confidence_interval=(0.25, 0.36),
    ),
}

# Regime-type modifiers: how regime type affects base rates
REGIME_MODIFIERS: dict[str, dict[str, float]] = {
    "democracy": {
        "military_buildup_to_invasion": 0.6,  # democracies less likely to invade
        "sanctions_threat_to_imposition": 1.2,  # more likely to follow through
        "ceasefire_holds_1yr": 1.3,
        "interstate_crisis_to_war": 0.5,
        "coup_attempt_success": 0.3,  # coups rare in stable democracies
    },
    "autocracy": {
        "military_buildup_to_invasion": 1.5,
        "sanctions_threat_to_imposition": 0.8,
        "ceasefire_holds_1yr": 0.7,
        "interstate_crisis_to_war": 1.8,
        "coup_attempt_success": 1.2,
    },
    "hybrid_regime": {
        "military_buildup_to_invasion": 1.2,
        "sanctions_threat_to_imposition": 1.0,
        "ceasefire_holds_1yr": 0.9,
        "interstate_crisis_to_war": 1.3,
        "coup_attempt_success": 1.5,
    },
}


class BaseRateEngine:
    """Engine for computing adjusted base rates given contextual conditions."""

    def __init__(self) -> None:
        self.rates = dict(HISTORICAL_BASE_RATES)
        self.regime_modifiers = dict(REGIME_MODIFIERS)

    def get_base_rate(self, event_type: str) -> BaseRateEstimate | None:
        """Get the raw historical base rate for an event type."""
        return self.rates.get(event_type)

    def get_adjusted_rate(
        self,
        event_type: str,
        regime_type: str | None = None,
        additional_factors: dict[str, float] | None = None,
    ) -> float:
        """Get base rate adjusted for regime type and contextual factors.

        Args:
            event_type: The type of event to estimate.
            regime_type: "democracy", "autocracy", or "hybrid_regime".
            additional_factors: Dict of factor_name -> multiplier for
                               additional contextual adjustments.

        Returns:
            Adjusted probability clamped to [0, 1].
        """
        rate_info = self.rates.get(event_type)
        if not rate_info:
            logger.warning("Unknown event type: %s", event_type)
            return 0.5  # uninformative prior

        adjusted = rate_info.base_rate

        # Apply regime modifier
        if regime_type:
            modifier = self.regime_modifiers.get(regime_type, {}).get(event_type, 1.0)
            adjusted *= modifier

        # Apply additional factors
        if additional_factors:
            for factor_name, multiplier in additional_factors.items():
                adjusted *= multiplier

        # Clamp to valid probability range
        return max(0.0, min(1.0, adjusted))

    def compute_bayesian_update(
        self,
        event_type: str,
        prior: float | None = None,
        evidence_likelihood_ratio: float = 1.0,
    ) -> float:
        """Apply Bayesian update to a base rate given new evidence.

        Args:
            event_type: The event type for the prior.
            prior: Override prior probability (default: use base rate).
            evidence_likelihood_ratio: P(evidence|event) / P(evidence|~event).

        Returns:
            Posterior probability.
        """
        if prior is None:
            rate = self.rates.get(event_type)
            prior = rate.base_rate if rate else 0.5

        # Bayes' theorem: P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|~H)*P(~H)]
        numerator = evidence_likelihood_ratio * prior
        denominator = numerator + (1.0 - prior)

        if denominator == 0:
            return prior

        return numerator / denominator

    def list_all_rates(self) -> list[dict[str, Any]]:
        """List all available base rates with metadata."""
        return [
            {
                "event_type": r.event_type,
                "description": r.description,
                "base_rate": r.base_rate,
                "sample_size": r.sample_size,
                "time_period": r.time_period,
                "confidence_interval": r.confidence_interval,
            }
            for r in self.rates.values()
        ]
