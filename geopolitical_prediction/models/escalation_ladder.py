"""Bayesian escalation ladder model.

Maps sequences of events to outcome probabilities using a Bayesian network
that models the escalation pathway from peace to conflict.

The ladder has discrete rungs:
  1. Normal relations
  2. Diplomatic tension
  3. Economic coercion (sanctions/trade war)
  4. Military posturing (buildup/exercises)
  5. Limited military action (border skirmish, proxy conflict)
  6. Full-scale military confrontation

The model estimates transition probabilities between rungs based on
observed evidence from ACLED, GDELT, and other data sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EscalationRung(IntEnum):
    """Discrete escalation levels in the ladder model."""

    NORMAL = 0
    DIPLOMATIC_TENSION = 1
    ECONOMIC_COERCION = 2
    MILITARY_POSTURING = 3
    LIMITED_MILITARY_ACTION = 4
    FULL_SCALE_CONFLICT = 5


# Prior transition probabilities (without evidence)
# P(next_rung | current_rung) — probability of escalating to the next rung
DEFAULT_TRANSITION_PROBS: dict[int, float] = {
    0: 0.10,  # Normal -> Diplomatic tension
    1: 0.15,  # Diplomatic tension -> Economic coercion
    2: 0.10,  # Economic coercion -> Military posturing
    3: 0.08,  # Military posturing -> Limited action
    4: 0.05,  # Limited action -> Full-scale conflict
}

# De-escalation probabilities (going down one rung)
DEFAULT_DEESCALATION_PROBS: dict[int, float] = {
    1: 0.30,  # Diplomatic tension -> Normal
    2: 0.20,  # Economic coercion -> Diplomatic tension
    3: 0.15,  # Military posturing -> Economic coercion
    4: 0.10,  # Limited action -> Military posturing
    5: 0.05,  # Full-scale conflict -> Limited action
}


@dataclass
class EscalationState:
    """Current state of the escalation ladder for a country pair."""

    country_a: str
    country_b: str
    current_rung: EscalationRung
    rung_probabilities: dict[int, float]  # probability of being at each rung
    transition_up_prob: float  # probability of escalating to next rung
    transition_down_prob: float  # probability of de-escalating
    time_at_current_rung: int  # days at current rung
    evidence_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationForecast:
    """Forward-looking escalation forecast."""

    country_a: str
    country_b: str
    current_rung: EscalationRung
    timestamp: datetime
    # Probabilities of reaching each rung within forecast horizon
    prob_diplomatic_tension: float
    prob_economic_coercion: float
    prob_military_posturing: float
    prob_limited_action: float
    prob_full_conflict: float
    # Derived
    prob_any_military: float  # P(rung >= 4)
    most_likely_outcome: str
    forecast_horizon_days: int
    confidence: float


class EscalationLadderModel:
    """Bayesian escalation ladder model for conflict probability estimation.

    Uses a hidden Markov model-like structure where:
    - States = escalation rungs
    - Observations = ACLED events, GDELT sentiment, economic indicators
    - Transitions = escalation/de-escalation probabilities

    Evidence updates transition probabilities via likelihood ratios.
    """

    def __init__(self) -> None:
        self.transition_up = dict(DEFAULT_TRANSITION_PROBS)
        self.transition_down = dict(DEFAULT_DEESCALATION_PROBS)

    def estimate_current_rung(
        self,
        conflict_score: float,
        sentiment_score: float,
        economic_coercion_active: bool,
        military_buildup_detected: bool,
        active_combat: bool,
    ) -> tuple[EscalationRung, dict[int, float]]:
        """Estimate current position on the escalation ladder.

        Uses a soft classification: returns probability distribution
        over all rungs rather than a hard assignment.
        """
        # Prior: uniform-ish with slight bias toward lower rungs
        probs = {
            0: 0.30,
            1: 0.25,
            2: 0.20,
            3: 0.15,
            4: 0.07,
            5: 0.03,
        }

        # Update based on conflict score (ACLED)
        if conflict_score > 0.7:
            probs[4] *= 3.0
            probs[5] *= 2.0
            probs[0] *= 0.1
        elif conflict_score > 0.4:
            probs[3] *= 2.5
            probs[4] *= 1.5
            probs[0] *= 0.3
        elif conflict_score > 0.2:
            probs[2] *= 2.0
            probs[3] *= 1.5
            probs[0] *= 0.5
        elif conflict_score < 0.05:
            probs[0] *= 3.0
            probs[4] *= 0.1
            probs[5] *= 0.05

        # Update based on sentiment (GDELT tone, lower = worse)
        if sentiment_score < -5:
            probs[1] *= 2.0
            probs[2] *= 1.8
            probs[0] *= 0.3
        elif sentiment_score < -2:
            probs[1] *= 1.5
            probs[0] *= 0.6

        # Economic coercion evidence
        if economic_coercion_active:
            probs[2] *= 3.0
            probs[0] *= 0.2
            probs[1] *= 0.5

        # Military buildup evidence
        if military_buildup_detected:
            probs[3] *= 4.0
            probs[4] *= 2.0
            probs[0] *= 0.1
            probs[1] *= 0.3

        # Active combat evidence
        if active_combat:
            probs[4] *= 5.0
            probs[5] *= 3.0
            probs[0] *= 0.01

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Most likely rung
        best_rung = max(probs, key=probs.get)
        return EscalationRung(best_rung), probs

    def compute_transition_probabilities(
        self,
        current_rung: EscalationRung,
        # Evidence inputs (each is a likelihood ratio multiplier)
        sentiment_trend: float = 1.0,  # >1 = deteriorating, <1 = improving
        conflict_trend: float = 1.0,  # >1 = escalating, <1 = de-escalating
        diplomatic_activity: float = 1.0,  # >1 = active diplomacy (de-escalation signal)
        domestic_pressure: float = 1.0,  # >1 = high pressure (escalation incentive)
        alliance_deterrence: float = 1.0,  # >1 = strong deterrence (de-escalation)
        economic_constraint: float = 1.0,  # >1 = high interdependence (de-escalation)
    ) -> tuple[float, float]:
        """Compute updated transition probabilities given evidence.

        Returns (prob_escalate, prob_de_escalate).
        """
        rung_val = int(current_rung)
        base_up = self.transition_up.get(rung_val, 0.01)
        base_down = self.transition_down.get(rung_val, 0.01)

        # Escalation factors
        escalation_lr = (
            sentiment_trend
            * conflict_trend
            * domestic_pressure
        )

        # De-escalation factors
        deescalation_lr = (
            diplomatic_activity
            * alliance_deterrence
            * economic_constraint
        )

        # Apply Bayesian updates
        up_odds = (base_up / (1 - base_up)) * escalation_lr / max(deescalation_lr, 0.01)
        prob_up = up_odds / (1 + up_odds)

        down_odds = (base_down / (1 - base_down)) * deescalation_lr / max(escalation_lr, 0.01)
        prob_down = down_odds / (1 + down_odds)

        # Ensure probabilities sum to <= 1
        total = prob_up + prob_down
        if total > 0.95:
            scale = 0.95 / total
            prob_up *= scale
            prob_down *= scale

        return (float(prob_up), float(prob_down))

    def forecast(
        self,
        country_a: str,
        country_b: str,
        current_rung: EscalationRung,
        rung_probs: dict[int, float],
        transition_up: float,
        transition_down: float,
        horizon_days: int = 90,
        time_steps: int = 90,
    ) -> EscalationForecast:
        """Generate forward-looking escalation forecast.

        Simulates the Markov chain forward to estimate the probability
        of reaching each escalation level within the forecast horizon.

        Uses matrix exponentiation for efficiency.
        """
        n_rungs = 6

        # Build transition matrix
        T = np.zeros((n_rungs, n_rungs))
        for i in range(n_rungs):
            up_p = transition_up if i == int(current_rung) else self.transition_up.get(i, 0.01)
            down_p = transition_down if i == int(current_rung) else self.transition_down.get(i, 0.01)

            if i < n_rungs - 1:
                T[i, i + 1] = up_p
            if i > 0:
                T[i, i - 1] = down_p
            T[i, i] = 1.0 - sum(T[i, :])  # remain at current rung

        # Ensure rows sum to 1
        for i in range(n_rungs):
            row_sum = T[i].sum()
            if row_sum > 0:
                T[i] /= row_sum

        # Initial state distribution
        state = np.array([rung_probs.get(i, 0) for i in range(n_rungs)])
        state = state / state.sum()

        # Propagate forward
        T_power = np.linalg.matrix_power(T, time_steps)
        final_state = state @ T_power

        # Compute "ever reached" probabilities (higher than final state)
        # Use simulation for more accurate cumulative estimates
        ever_reached = np.zeros(n_rungs)
        current_state = state.copy()
        for _ in range(time_steps):
            current_state = current_state @ T
            ever_reached = np.maximum(ever_reached, current_state)

        prob_any_military = float(ever_reached[4] + ever_reached[5])

        # Most likely outcome
        outcomes = [
            "normal_relations",
            "diplomatic_tension",
            "economic_coercion",
            "military_posturing",
            "limited_military_action",
            "full_scale_conflict",
        ]
        most_likely_idx = int(np.argmax(final_state))

        # Confidence based on how peaked the distribution is
        entropy = -np.sum(final_state * np.log(final_state + 1e-10))
        max_entropy = np.log(n_rungs)
        confidence = 1.0 - (entropy / max_entropy)

        return EscalationForecast(
            country_a=country_a,
            country_b=country_b,
            current_rung=current_rung,
            timestamp=datetime.utcnow(),
            prob_diplomatic_tension=float(ever_reached[1]),
            prob_economic_coercion=float(ever_reached[2]),
            prob_military_posturing=float(ever_reached[3]),
            prob_limited_action=float(ever_reached[4]),
            prob_full_conflict=float(ever_reached[5]),
            prob_any_military=prob_any_military,
            most_likely_outcome=outcomes[most_likely_idx],
            forecast_horizon_days=horizon_days,
            confidence=float(confidence),
        )
