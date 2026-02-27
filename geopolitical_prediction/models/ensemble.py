"""Ensemble probability engine combining multiple signal sources.

Produces final probability estimates for geopolitical prediction market
questions by combining:
  - GDELT sentiment trajectory
  - ACLED conflict event frequency
  - Economic indicators
  - Escalation ladder model
  - Historical analog matching
  - Prediction market prices (as a calibration reference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from geopolitical_prediction.config import ModelConfig, get_settings
from geopolitical_prediction.features.base_rates import BaseRateEngine
from geopolitical_prediction.models.escalation_ladder import (
    EscalationForecast,
    EscalationLadderModel,
    EscalationRung,
)
from geopolitical_prediction.models.analog_matching import AnalogForecast, AnalogMatcher

logger = logging.getLogger(__name__)


@dataclass
class MarketQuestion:
    """A geopolitical prediction market question."""

    question_id: str
    question_text: str
    question_type: str  # invasion, sanctions, trade_deal, coup, ceasefire, etc.
    country_a: str
    country_b: str | None
    resolution_date: datetime | None
    base_rate_key: str  # maps to BaseRateEngine event types


@dataclass
class ProbabilityEstimate:
    """Final probability estimate for a market question."""

    question: MarketQuestion
    timestamp: datetime
    # Final estimate
    probability: float
    confidence: float
    # Component estimates
    base_rate: float
    escalation_model_prob: float
    analog_model_prob: float
    sentiment_signal: float  # -1 to 1, negative = bearish
    conflict_signal: float  # 0 to 1
    economic_signal: float  # 0 to 1, higher = more constraint on escalation
    market_price: float | None  # current prediction market price
    # Model vs market divergence
    model_market_divergence: float | None
    # Metadata
    signals_used: list[str]
    signal_weights: dict[str, float]
    reasoning: list[str]


@dataclass
class EnsembleWeights:
    """Adaptive weights for ensemble components."""

    base_rate: float = 0.15
    escalation_model: float = 0.25
    analog_model: float = 0.15
    sentiment: float = 0.15
    conflict_events: float = 0.15
    economic: float = 0.10
    market_price: float = 0.05


class EnsembleProbabilityEngine:
    """Main probability engine that produces forecasts for market questions.

    Combines multiple signal sources with adaptive weighting to produce
    calibrated probability estimates.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or get_settings().model
        self.base_rate_engine = BaseRateEngine()
        self.escalation_model = EscalationLadderModel()
        self.analog_matcher = AnalogMatcher()
        self.weights = EnsembleWeights()
        self._calibration_history: list[dict[str, Any]] = []

    def estimate_probability(
        self,
        question: MarketQuestion,
        # Signal inputs
        conflict_score: float = 0.0,
        sentiment_tone: float = 0.0,
        sentiment_trend: float = 0.0,
        economic_interdependence: float = 0.0,
        military_buildup: bool = False,
        active_combat: bool = False,
        economic_coercion: bool = False,
        domestic_pressure: float = 0.0,
        alliance_deterrence: float = 0.0,
        market_price: float | None = None,
        # Analog matching inputs
        trigger_type: str = "territorial",
        regime_type_a: str = "democracy",
        regime_type_b: str = "democracy",
        nuclear_parties: bool = False,
        alliance_involvement: bool = False,
        prior_conflicts: int = 0,
    ) -> ProbabilityEstimate:
        """Produce a probability estimate for a market question.

        Combines all available signals into a single calibrated estimate.
        """
        signals_used = []
        reasoning = []

        # 1. Base rate
        base_rate = self.base_rate_engine.get_adjusted_rate(
            question.base_rate_key,
            regime_type=regime_type_a,
        )
        signals_used.append("base_rate")
        reasoning.append(f"Historical base rate for {question.base_rate_key}: {base_rate:.2%}")

        # 2. Escalation ladder model
        current_rung, rung_probs = self.escalation_model.estimate_current_rung(
            conflict_score=conflict_score,
            sentiment_score=sentiment_tone,
            economic_coercion_active=economic_coercion,
            military_buildup_detected=military_buildup,
            active_combat=active_combat,
        )

        escalation_prob = self._map_escalation_to_question(
            question, current_rung, rung_probs
        )
        signals_used.append("escalation_model")
        reasoning.append(
            f"Escalation model: current rung={current_rung.name}, "
            f"mapped probability={escalation_prob:.2%}"
        )

        # 3. Analog matching
        analogs = self.analog_matcher.find_analogs(
            trigger=trigger_type,
            regime_type_a=regime_type_a,
            regime_type_b=regime_type_b,
            nuclear_parties=nuclear_parties,
            alliance_involvement=alliance_involvement,
            economic_interdependence=economic_interdependence,
            prior_conflicts=prior_conflicts,
            initial_escalation_level=int(current_rung),
        )
        analog_forecast = self.analog_matcher.forecast_from_analogs(analogs)
        analog_prob = self._map_analog_to_question(question, analog_forecast)
        signals_used.append("analog_model")
        if analogs:
            top_analog = analogs[0].crisis.name
            reasoning.append(
                f"Top historical analog: {top_analog} "
                f"(similarity={analogs[0].similarity_score:.2f}), "
                f"mapped probability={analog_prob:.2%}"
            )

        # 4. Sentiment signal (-1 to 1, converted to probability modifier)
        sentiment_signal = self._sentiment_to_signal(sentiment_tone, sentiment_trend)
        signals_used.append("sentiment")
        reasoning.append(
            f"GDELT sentiment signal: {sentiment_signal:+.2f} "
            f"(tone={sentiment_tone:.1f}, trend={sentiment_trend:+.2f})"
        )

        # 5. Conflict events signal
        conflict_signal = min(conflict_score, 1.0)
        signals_used.append("conflict_events")
        reasoning.append(f"ACLED conflict signal: {conflict_signal:.2f}")

        # 6. Economic constraint signal
        economic_signal = economic_interdependence
        signals_used.append("economic")
        reasoning.append(
            f"Economic interdependence constraint: {economic_signal:.2f}"
        )

        # 7. Market price (if available, use as weak anchor)
        market_component = market_price if market_price is not None else base_rate
        if market_price is not None:
            signals_used.append("market_price")
            reasoning.append(f"Current market price: {market_price:.2%}")

        # Ensemble combination
        w = self.weights
        raw_prob = (
            w.base_rate * base_rate
            + w.escalation_model * escalation_prob
            + w.analog_model * analog_prob
            + w.sentiment * self._signal_to_prob(sentiment_signal, base_rate)
            + w.conflict_events * conflict_signal
            + w.economic * (1.0 - economic_signal)  # high interdependence = lower escalation prob
            + w.market_price * market_component
        )

        # Clamp and apply domestic pressure adjustment
        if domestic_pressure > 0.5:
            raw_prob *= 1.0 + (domestic_pressure - 0.5) * 0.4
            reasoning.append(
                f"Domestic pressure adjustment: +{(domestic_pressure - 0.5) * 0.4:.1%}"
            )

        # Alliance deterrence adjustment
        if alliance_deterrence > 0.5:
            raw_prob *= 1.0 - (alliance_deterrence - 0.5) * 0.3
            reasoning.append(
                f"Alliance deterrence adjustment: -{(alliance_deterrence - 0.5) * 0.3:.1%}"
            )

        probability = max(0.01, min(0.99, raw_prob))

        # Confidence: based on signal availability and agreement
        signal_values = [base_rate, escalation_prob, analog_prob, conflict_signal]
        signal_std = float(np.std(signal_values))
        confidence = max(0.1, min(0.9, 1.0 - signal_std * 2))

        # Model-market divergence
        divergence = None
        if market_price is not None:
            divergence = probability - market_price
            if abs(divergence) > 0.15:
                reasoning.append(
                    f"Significant model-market divergence: {divergence:+.2%}"
                )

        return ProbabilityEstimate(
            question=question,
            timestamp=datetime.utcnow(),
            probability=probability,
            confidence=confidence,
            base_rate=base_rate,
            escalation_model_prob=escalation_prob,
            analog_model_prob=analog_prob,
            sentiment_signal=sentiment_signal,
            conflict_signal=conflict_signal,
            economic_signal=economic_signal,
            market_price=market_price,
            model_market_divergence=divergence,
            signals_used=signals_used,
            signal_weights={
                "base_rate": w.base_rate,
                "escalation_model": w.escalation_model,
                "analog_model": w.analog_model,
                "sentiment": w.sentiment,
                "conflict_events": w.conflict_events,
                "economic": w.economic,
                "market_price": w.market_price,
            },
            reasoning=reasoning,
        )

    def _map_escalation_to_question(
        self,
        question: MarketQuestion,
        current_rung: EscalationRung,
        rung_probs: dict[int, float],
    ) -> float:
        """Map escalation ladder state to question-specific probability."""
        qtype = question.question_type

        if qtype == "invasion":
            # Invasion maps to rungs 4-5
            return rung_probs.get(4, 0) + rung_probs.get(5, 0)
        elif qtype == "sanctions":
            # Sanctions maps to rungs 2+
            return sum(rung_probs.get(i, 0) for i in range(2, 6))
        elif qtype == "trade_deal":
            # Trade deal is more likely at lower rungs
            return rung_probs.get(0, 0) * 0.4 + rung_probs.get(1, 0) * 0.2
        elif qtype == "ceasefire":
            # Ceasefire probability depends on being in active conflict
            if current_rung >= EscalationRung.LIMITED_MILITARY_ACTION:
                return 0.3  # some chance of ceasefire during conflict
            return 0.1
        elif qtype == "coup":
            return rung_probs.get(3, 0) * 0.3 + rung_probs.get(4, 0) * 0.2
        else:
            return 0.5  # default: uninformative

    def _map_analog_to_question(
        self,
        question: MarketQuestion,
        forecast: AnalogForecast,
    ) -> float:
        """Map analog forecast outcomes to question-specific probability."""
        qtype = question.question_type
        probs = forecast.outcome_probabilities

        if qtype == "invasion":
            return probs.get("war", 0) + probs.get("limited_conflict", 0) * 0.3
        elif qtype == "sanctions":
            return probs.get("sanctions", 0) + probs.get("war", 0) * 0.5
        elif qtype == "trade_deal":
            return probs.get("negotiated_settlement", 0) + probs.get("de_escalation", 0) * 0.3
        elif qtype == "ceasefire":
            return probs.get("de_escalation", 0) + probs.get("negotiated_settlement", 0) * 0.5
        else:
            return 0.5

    @staticmethod
    def _sentiment_to_signal(tone: float, trend: float) -> float:
        """Convert GDELT tone and trend to a -1 to 1 signal.

        Negative signal = deteriorating/hostile (increases conflict probability)
        Positive signal = improving/cooperative (decreases conflict probability)
        """
        # Tone ranges from about -10 to +10 typically
        tone_signal = max(-1, min(1, tone / 10.0))
        # Trend is slope; negative trend = deteriorating
        trend_signal = max(-1, min(1, trend * 5))

        return 0.6 * tone_signal + 0.4 * trend_signal

    @staticmethod
    def _signal_to_prob(signal: float, base_rate: float) -> float:
        """Convert a -1 to 1 signal to a probability centered on base rate."""
        # Shift base rate up or down based on signal
        return max(0.01, min(0.99, base_rate + signal * 0.3))

    def record_outcome(
        self,
        question_id: str,
        predicted_prob: float,
        actual_outcome: bool,
    ) -> None:
        """Record a resolved prediction for calibration tracking."""
        self._calibration_history.append(
            {
                "question_id": question_id,
                "predicted": predicted_prob,
                "actual": 1.0 if actual_outcome else 0.0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def compute_calibration(self, n_bins: int = 10) -> dict[str, Any]:
        """Compute calibration metrics from historical predictions.

        Returns Brier score, calibration curve data, and reliability diagram bins.
        """
        if not self._calibration_history:
            return {"brier_score": None, "n_predictions": 0}

        predictions = np.array([h["predicted"] for h in self._calibration_history])
        actuals = np.array([h["actual"] for h in self._calibration_history])

        # Brier score
        brier = float(np.mean((predictions - actuals) ** 2))

        # Calibration bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if mask.sum() > 0:
                bin_data.append(
                    {
                        "bin_center": float((bins[i] + bins[i + 1]) / 2),
                        "mean_predicted": float(predictions[mask].mean()),
                        "mean_actual": float(actuals[mask].mean()),
                        "count": int(mask.sum()),
                    }
                )

        return {
            "brier_score": brier,
            "n_predictions": len(predictions),
            "calibration_bins": bin_data,
        }
