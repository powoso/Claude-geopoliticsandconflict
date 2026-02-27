"""Historical analog matching for crisis resolution prediction.

Finds historically similar crises and tracks how they resolved,
providing empirical base rates adjusted for similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HistoricalCrisis:
    """A coded historical crisis with features and outcome."""

    name: str
    year: int
    # Feature vector
    parties: list[str]
    trigger: str  # territorial, ethnic, resource, ideological
    regime_type_a: str  # democracy, autocracy, hybrid
    regime_type_b: str
    nuclear_parties: bool  # whether nuclear powers are directly involved
    alliance_involvement: bool
    economic_interdependence: float  # 0-1
    prior_conflicts: int  # number of previous crises between parties
    initial_escalation_level: int  # 0-5
    peak_escalation_level: int
    # Outcome
    outcome: str  # war, limited_conflict, sanctions, negotiated_settlement, status_quo, de_escalation
    duration_months: int
    fatalities: int
    resolution_mechanism: str  # bilateral_negotiation, mediation, un_intervention, exhaustion, deterrence


@dataclass
class AnalogMatch:
    """A historical crisis matched as an analog to a current situation."""

    crisis: HistoricalCrisis
    similarity_score: float  # 0-1
    matching_features: list[str]
    differing_features: list[str]


@dataclass
class AnalogForecast:
    """Forecast based on historical analogs."""

    current_crisis_description: str
    top_analogs: list[AnalogMatch]
    outcome_probabilities: dict[str, float]
    weighted_duration_months: float
    confidence: float


# Historical crisis database (representative sample for demonstration)
CRISIS_DATABASE: list[HistoricalCrisis] = [
    HistoricalCrisis(
        name="Cuban Missile Crisis",
        year=1962,
        parties=["United States", "Soviet Union"],
        trigger="territorial",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=True,
        alliance_involvement=True,
        economic_interdependence=0.05,
        prior_conflicts=3,
        initial_escalation_level=4,
        peak_escalation_level=5,
        outcome="de_escalation",
        duration_months=1,
        fatalities=0,
        resolution_mechanism="bilateral_negotiation",
    ),
    HistoricalCrisis(
        name="Falklands War",
        year=1982,
        parties=["United Kingdom", "Argentina"],
        trigger="territorial",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=False,
        alliance_involvement=False,
        economic_interdependence=0.1,
        prior_conflicts=1,
        initial_escalation_level=3,
        peak_escalation_level=5,
        outcome="war",
        duration_months=3,
        fatalities=907,
        resolution_mechanism="exhaustion",
    ),
    HistoricalCrisis(
        name="Gulf War 1990-91",
        year=1990,
        parties=["United States", "Iraq"],
        trigger="territorial",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=False,
        alliance_involvement=True,
        economic_interdependence=0.15,
        prior_conflicts=0,
        initial_escalation_level=3,
        peak_escalation_level=5,
        outcome="war",
        duration_months=7,
        fatalities=30000,
        resolution_mechanism="un_intervention",
    ),
    HistoricalCrisis(
        name="Kargil War",
        year=1999,
        parties=["India", "Pakistan"],
        trigger="territorial",
        regime_type_a="democracy",
        regime_type_b="democracy",
        nuclear_parties=True,
        alliance_involvement=False,
        economic_interdependence=0.05,
        prior_conflicts=3,
        initial_escalation_level=4,
        peak_escalation_level=4,
        outcome="limited_conflict",
        duration_months=3,
        fatalities=1200,
        resolution_mechanism="bilateral_negotiation",
    ),
    HistoricalCrisis(
        name="Russia-Georgia War",
        year=2008,
        parties=["Russia", "Georgia"],
        trigger="territorial",
        regime_type_a="autocracy",
        regime_type_b="hybrid",
        nuclear_parties=False,
        alliance_involvement=False,
        economic_interdependence=0.1,
        prior_conflicts=2,
        initial_escalation_level=3,
        peak_escalation_level=5,
        outcome="war",
        duration_months=1,
        fatalities=850,
        resolution_mechanism="mediation",
    ),
    HistoricalCrisis(
        name="Crimea Annexation",
        year=2014,
        parties=["Russia", "Ukraine"],
        trigger="territorial",
        regime_type_a="autocracy",
        regime_type_b="hybrid",
        nuclear_parties=False,
        alliance_involvement=False,
        economic_interdependence=0.2,
        prior_conflicts=0,
        initial_escalation_level=3,
        peak_escalation_level=4,
        outcome="limited_conflict",
        duration_months=2,
        fatalities=100,
        resolution_mechanism="exhaustion",
    ),
    HistoricalCrisis(
        name="Russia-Ukraine War 2022",
        year=2022,
        parties=["Russia", "Ukraine"],
        trigger="territorial",
        regime_type_a="autocracy",
        regime_type_b="democracy",
        nuclear_parties=False,
        alliance_involvement=True,
        economic_interdependence=0.15,
        prior_conflicts=2,
        initial_escalation_level=4,
        peak_escalation_level=5,
        outcome="war",
        duration_months=36,
        fatalities=500000,
        resolution_mechanism="exhaustion",
    ),
    HistoricalCrisis(
        name="Iran Sanctions 2012",
        year=2012,
        parties=["United States", "Iran"],
        trigger="ideological",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=False,
        alliance_involvement=True,
        economic_interdependence=0.02,
        prior_conflicts=2,
        initial_escalation_level=2,
        peak_escalation_level=2,
        outcome="sanctions",
        duration_months=36,
        fatalities=0,
        resolution_mechanism="bilateral_negotiation",
    ),
    HistoricalCrisis(
        name="US-China Trade War 2018",
        year=2018,
        parties=["United States", "China"],
        trigger="resource",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=True,
        alliance_involvement=False,
        economic_interdependence=0.8,
        prior_conflicts=1,
        initial_escalation_level=2,
        peak_escalation_level=2,
        outcome="negotiated_settlement",
        duration_months=24,
        fatalities=0,
        resolution_mechanism="bilateral_negotiation",
    ),
    HistoricalCrisis(
        name="North Korea Nuclear Crises",
        year=2017,
        parties=["United States", "North Korea"],
        trigger="ideological",
        regime_type_a="democracy",
        regime_type_b="autocracy",
        nuclear_parties=True,
        alliance_involvement=True,
        economic_interdependence=0.0,
        prior_conflicts=5,
        initial_escalation_level=3,
        peak_escalation_level=3,
        outcome="status_quo",
        duration_months=12,
        fatalities=0,
        resolution_mechanism="deterrence",
    ),
    HistoricalCrisis(
        name="Taiwan Strait Crisis 1996",
        year=1996,
        parties=["China", "Taiwan"],
        trigger="territorial",
        regime_type_a="autocracy",
        regime_type_b="democracy",
        nuclear_parties=True,
        alliance_involvement=True,
        economic_interdependence=0.3,
        prior_conflicts=2,
        initial_escalation_level=3,
        peak_escalation_level=3,
        outcome="de_escalation",
        duration_months=3,
        fatalities=0,
        resolution_mechanism="deterrence",
    ),
    HistoricalCrisis(
        name="Kosovo War",
        year=1999,
        parties=["NATO", "Serbia"],
        trigger="ethnic",
        regime_type_a="democracy",
        regime_type_b="hybrid",
        nuclear_parties=False,
        alliance_involvement=True,
        economic_interdependence=0.05,
        prior_conflicts=1,
        initial_escalation_level=3,
        peak_escalation_level=5,
        outcome="war",
        duration_months=3,
        fatalities=13000,
        resolution_mechanism="un_intervention",
    ),
]


def _crisis_to_feature_vector(crisis: HistoricalCrisis) -> np.ndarray:
    """Convert a crisis to a numerical feature vector for similarity computation."""
    trigger_map = {"territorial": 0, "ethnic": 1, "resource": 2, "ideological": 3}
    regime_map = {"democracy": 0, "autocracy": 1, "hybrid": 0.5}

    return np.array([
        trigger_map.get(crisis.trigger, 2),
        regime_map.get(crisis.regime_type_a, 0.5),
        regime_map.get(crisis.regime_type_b, 0.5),
        float(crisis.nuclear_parties),
        float(crisis.alliance_involvement),
        crisis.economic_interdependence,
        min(crisis.prior_conflicts / 5.0, 1.0),
        crisis.initial_escalation_level / 5.0,
    ])


def compute_similarity(crisis_a: HistoricalCrisis, features_b: np.ndarray) -> float:
    """Compute similarity between a historical crisis and current features.

    Uses weighted Euclidean distance converted to a 0-1 similarity score.
    """
    vec_a = _crisis_to_feature_vector(crisis_a)

    # Feature weights (importance of each dimension)
    weights = np.array([
        0.15,  # trigger type
        0.10,  # regime type A
        0.10,  # regime type B
        0.15,  # nuclear involvement
        0.10,  # alliance involvement
        0.15,  # economic interdependence
        0.10,  # prior conflicts
        0.15,  # initial escalation level
    ])

    distance = np.sqrt(np.sum(weights * (vec_a - features_b) ** 2))
    max_distance = np.sqrt(np.sum(weights))  # max possible distance

    return float(1.0 - (distance / max_distance))


class AnalogMatcher:
    """Find and analyze historical analogs for current crises."""

    def __init__(self, crisis_db: list[HistoricalCrisis] | None = None):
        self.crisis_db = crisis_db or CRISIS_DATABASE

    def find_analogs(
        self,
        trigger: str,
        regime_type_a: str,
        regime_type_b: str,
        nuclear_parties: bool,
        alliance_involvement: bool,
        economic_interdependence: float,
        prior_conflicts: int,
        initial_escalation_level: int,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[AnalogMatch]:
        """Find top-K most similar historical crises."""
        trigger_map = {"territorial": 0, "ethnic": 1, "resource": 2, "ideological": 3}
        regime_map = {"democracy": 0, "autocracy": 1, "hybrid": 0.5}

        current_features = np.array([
            trigger_map.get(trigger, 2),
            regime_map.get(regime_type_a, 0.5),
            regime_map.get(regime_type_b, 0.5),
            float(nuclear_parties),
            float(alliance_involvement),
            economic_interdependence,
            min(prior_conflicts / 5.0, 1.0),
            initial_escalation_level / 5.0,
        ])

        matches = []
        for crisis in self.crisis_db:
            sim = compute_similarity(crisis, current_features)
            if sim >= min_similarity:
                # Determine matching/differing features
                matching = []
                differing = []

                if trigger == crisis.trigger:
                    matching.append("trigger_type")
                else:
                    differing.append("trigger_type")

                if regime_type_a == crisis.regime_type_a:
                    matching.append("regime_type_a")
                else:
                    differing.append("regime_type_a")

                if nuclear_parties == crisis.nuclear_parties:
                    matching.append("nuclear_involvement")
                else:
                    differing.append("nuclear_involvement")

                if alliance_involvement == crisis.alliance_involvement:
                    matching.append("alliance_involvement")
                else:
                    differing.append("alliance_involvement")

                matches.append(
                    AnalogMatch(
                        crisis=crisis,
                        similarity_score=sim,
                        matching_features=matching,
                        differing_features=differing,
                    )
                )

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:top_k]

    def forecast_from_analogs(
        self,
        analogs: list[AnalogMatch],
        description: str = "",
    ) -> AnalogForecast:
        """Generate probabilistic forecast from matched analogs.

        Weights outcomes by similarity score to produce probability
        distribution over possible outcomes.
        """
        if not analogs:
            return AnalogForecast(
                current_crisis_description=description,
                top_analogs=[],
                outcome_probabilities={
                    "war": 0.1,
                    "limited_conflict": 0.15,
                    "sanctions": 0.15,
                    "negotiated_settlement": 0.2,
                    "status_quo": 0.2,
                    "de_escalation": 0.2,
                },
                weighted_duration_months=12,
                confidence=0.1,
            )

        # Similarity-weighted outcome counts
        outcome_weights: dict[str, float] = {}
        total_weight = 0.0
        weighted_duration = 0.0

        for match in analogs:
            w = match.similarity_score
            outcome = match.crisis.outcome
            outcome_weights[outcome] = outcome_weights.get(outcome, 0) + w
            total_weight += w
            weighted_duration += w * match.crisis.duration_months

        # Normalize to probabilities
        outcome_probs = {k: v / total_weight for k, v in outcome_weights.items()}

        # Ensure all outcomes are represented
        for outcome in ["war", "limited_conflict", "sanctions", "negotiated_settlement",
                        "status_quo", "de_escalation"]:
            if outcome not in outcome_probs:
                outcome_probs[outcome] = 0.0

        # Confidence based on number and quality of analogs
        avg_similarity = total_weight / len(analogs)
        confidence = min(avg_similarity * (len(analogs) / 5.0), 0.9)

        return AnalogForecast(
            current_crisis_description=description,
            top_analogs=analogs,
            outcome_probabilities=outcome_probs,
            weighted_duration_months=weighted_duration / total_weight if total_weight > 0 else 12,
            confidence=confidence,
        )
