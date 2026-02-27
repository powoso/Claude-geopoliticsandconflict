"""Tests for probability engine models."""

from datetime import datetime

import pytest

from geopolitical_prediction.models.escalation_ladder import (
    EscalationForecast,
    EscalationLadderModel,
    EscalationRung,
)
from geopolitical_prediction.models.analog_matching import (
    AnalogMatcher,
    CRISIS_DATABASE,
)
from geopolitical_prediction.models.ensemble import (
    EnsembleProbabilityEngine,
    MarketQuestion,
)


class TestEscalationLadder:
    @pytest.fixture
    def model(self):
        return EscalationLadderModel()

    def test_estimate_rung_peaceful(self, model):
        rung, probs = model.estimate_current_rung(
            conflict_score=0.0,
            sentiment_score=5.0,
            economic_coercion_active=False,
            military_buildup_detected=False,
            active_combat=False,
        )
        assert rung == EscalationRung.NORMAL
        assert probs[0] > probs[5]  # Normal more likely than full conflict

    def test_estimate_rung_crisis(self, model):
        rung, probs = model.estimate_current_rung(
            conflict_score=0.8,
            sentiment_score=-8.0,
            economic_coercion_active=True,
            military_buildup_detected=True,
            active_combat=True,
        )
        assert rung.value >= 3  # Should be at military level or above
        assert probs[4] + probs[5] > probs[0]  # Military more likely than normal

    def test_transition_probabilities(self, model):
        up, down = model.compute_transition_probabilities(
            current_rung=EscalationRung.DIPLOMATIC_TENSION,
            sentiment_trend=2.0,  # deteriorating
            conflict_trend=1.5,
        )
        assert 0 <= up <= 1
        assert 0 <= down <= 1
        assert up + down <= 1

    def test_deterrence_reduces_escalation(self, model):
        up_no_deterrence, _ = model.compute_transition_probabilities(
            current_rung=EscalationRung.MILITARY_POSTURING,
        )
        up_with_deterrence, _ = model.compute_transition_probabilities(
            current_rung=EscalationRung.MILITARY_POSTURING,
            alliance_deterrence=3.0,
        )
        assert up_with_deterrence < up_no_deterrence

    def test_forecast(self, model):
        rung_probs = {0: 0.05, 1: 0.1, 2: 0.15, 3: 0.4, 4: 0.2, 5: 0.1}
        forecast = model.forecast(
            country_a="Russia",
            country_b="Ukraine",
            current_rung=EscalationRung.MILITARY_POSTURING,
            rung_probs=rung_probs,
            transition_up=0.1,
            transition_down=0.05,
            horizon_days=90,
        )
        assert isinstance(forecast, EscalationForecast)
        assert 0 <= forecast.prob_full_conflict <= 1
        assert 0 <= forecast.prob_any_military <= 1
        assert 0 < forecast.confidence < 1
        assert forecast.forecast_horizon_days == 90


class TestAnalogMatching:
    @pytest.fixture
    def matcher(self):
        return AnalogMatcher()

    def test_find_analogs_territorial(self, matcher):
        analogs = matcher.find_analogs(
            trigger="territorial",
            regime_type_a="autocracy",
            regime_type_b="democracy",
            nuclear_parties=False,
            alliance_involvement=False,
            economic_interdependence=0.15,
            prior_conflicts=1,
            initial_escalation_level=3,
        )
        assert len(analogs) > 0
        assert all(a.similarity_score >= 0.3 for a in analogs)
        # Should find Russia-Georgia, Crimea, etc.

    def test_find_analogs_nuclear_crisis(self, matcher):
        analogs = matcher.find_analogs(
            trigger="territorial",
            regime_type_a="democracy",
            regime_type_b="autocracy",
            nuclear_parties=True,
            alliance_involvement=True,
            economic_interdependence=0.3,
            prior_conflicts=2,
            initial_escalation_level=3,
        )
        assert len(analogs) > 0
        # Cuban Missile Crisis and Taiwan Strait should be top matches

    def test_forecast_from_analogs(self, matcher):
        analogs = matcher.find_analogs(
            trigger="territorial",
            regime_type_a="autocracy",
            regime_type_b="hybrid",
            nuclear_parties=False,
            alliance_involvement=False,
            economic_interdependence=0.1,
            prior_conflicts=1,
            initial_escalation_level=3,
        )
        forecast = matcher.forecast_from_analogs(analogs, "Test crisis")
        assert sum(forecast.outcome_probabilities.values()) == pytest.approx(1.0, abs=0.01)
        assert forecast.weighted_duration_months > 0
        assert 0 < forecast.confidence < 1

    def test_empty_analogs_forecast(self, matcher):
        forecast = matcher.forecast_from_analogs([], "No matches")
        assert forecast.confidence == 0.1
        assert sum(forecast.outcome_probabilities.values()) == pytest.approx(1.0, abs=0.01)


class TestEnsembleEngine:
    @pytest.fixture
    def engine(self):
        return EnsembleProbabilityEngine()

    @pytest.fixture
    def invasion_question(self):
        return MarketQuestion(
            question_id="test_invasion",
            question_text="Will X invade Y by 2025?",
            question_type="invasion",
            country_a="TestCountryA",
            country_b="TestCountryB",
            resolution_date=datetime(2025, 12, 31),
            base_rate_key="military_buildup_to_invasion",
        )

    @pytest.fixture
    def sanctions_question(self):
        return MarketQuestion(
            question_id="test_sanctions",
            question_text="Will sanctions be imposed on X?",
            question_type="sanctions",
            country_a="TestCountryA",
            country_b="TestCountryB",
            resolution_date=datetime(2025, 12, 31),
            base_rate_key="sanctions_threat_to_imposition",
        )

    def test_basic_estimate(self, engine, invasion_question):
        estimate = engine.estimate_probability(invasion_question)
        assert 0 < estimate.probability < 1
        assert 0 < estimate.confidence < 1
        assert len(estimate.signals_used) > 0
        assert len(estimate.reasoning) > 0

    def test_high_conflict_increases_probability(self, engine, invasion_question):
        low = engine.estimate_probability(
            invasion_question,
            conflict_score=0.1,
            sentiment_tone=2.0,
        )
        high = engine.estimate_probability(
            invasion_question,
            conflict_score=0.9,
            sentiment_tone=-8.0,
            military_buildup=True,
            active_combat=True,
        )
        assert high.probability > low.probability

    def test_sanctions_question(self, engine, sanctions_question):
        estimate = engine.estimate_probability(
            sanctions_question,
            conflict_score=0.3,
            economic_coercion=True,
        )
        assert 0 < estimate.probability < 1

    def test_market_price_divergence(self, engine, invasion_question):
        estimate = engine.estimate_probability(
            invasion_question,
            market_price=0.1,
        )
        assert estimate.model_market_divergence is not None
        assert estimate.market_price == 0.1

    def test_calibration_tracking(self, engine):
        engine.record_outcome("q1", 0.7, True)
        engine.record_outcome("q2", 0.3, False)
        engine.record_outcome("q3", 0.8, True)

        cal = engine.compute_calibration()
        assert cal["n_predictions"] == 3
        assert cal["brier_score"] is not None
        assert 0 <= cal["brier_score"] <= 1

    def test_economic_constraint_effect(self, engine, invasion_question):
        low_interdependence = engine.estimate_probability(
            invasion_question,
            economic_interdependence=0.1,
        )
        high_interdependence = engine.estimate_probability(
            invasion_question,
            economic_interdependence=0.9,
        )
        # High economic interdependence should reduce invasion probability
        assert high_interdependence.probability < low_interdependence.probability
