"""Tests for feature engineering modules."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from geopolitical_prediction.features.conflict_scoring import (
    EscalationScore,
    compute_escalation_score,
    compute_rolling_event_rate,
    compute_spatial_spread,
    compute_actor_diversity,
    compute_civilian_targeting_ratio,
)
from geopolitical_prediction.features.sentiment import (
    extract_sentiment_features,
    compute_sentiment_trajectory,
    bilateral_sentiment_divergence,
)
from geopolitical_prediction.features.base_rates import (
    BaseRateEngine,
    HISTORICAL_BASE_RATES,
)
from geopolitical_prediction.features.economic import (
    compute_interdependence_score,
    compute_escalation_constraint,
    EconomicAnalyzer,
)
from geopolitical_prediction.features.political import (
    DomesticPoliticalState,
    compute_diversionary_risk,
)
from geopolitical_prediction.features.alliance import AllianceNetworkAnalyzer


class TestConflictScoring:
    @pytest.fixture
    def sample_events_df(self):
        now = datetime.utcnow()
        data = []
        for i in range(100):
            data.append(
                {
                    "event_id": f"EVT{i}",
                    "event_date": now - timedelta(days=i % 90),
                    "event_type": np.random.choice([
                        "Battles",
                        "Explosions/Remote violence",
                        "Violence against civilians",
                        "Protests",
                    ]),
                    "actor1": f"Actor{i % 5}",
                    "actor2": f"Actor{(i + 2) % 5}",
                    "country": "TestCountry",
                    "latitude": 48.0 + np.random.uniform(-2, 2),
                    "longitude": 38.0 + np.random.uniform(-2, 2),
                    "fatalities": np.random.randint(0, 20),
                    "weighted_severity": np.random.uniform(0.2, 2.0),
                }
            )
        df = pd.DataFrame(data)
        df["event_date"] = pd.to_datetime(df["event_date"])
        return df

    def test_compute_escalation_score(self, sample_events_df):
        score = compute_escalation_score(sample_events_df, "TestCountry")
        assert isinstance(score, EscalationScore)
        assert 0 <= score.composite_score <= 1
        assert score.trend in ("escalating", "stable", "de_escalating", "insufficient_data")
        assert all(0 <= v <= 1 for v in score.components.values())

    def test_spatial_spread_concentrated(self):
        df = pd.DataFrame(
            {"latitude": [48.0] * 10, "longitude": [38.0] * 10}
        )
        spread = compute_spatial_spread(df)
        assert spread == 0.0

    def test_spatial_spread_dispersed(self):
        df = pd.DataFrame(
            {
                "latitude": np.linspace(30, 50, 20),
                "longitude": np.linspace(20, 50, 20),
            }
        )
        spread = compute_spatial_spread(df)
        assert spread > 0.5

    def test_actor_diversity_single(self):
        df = pd.DataFrame({"actor1": ["A"] * 5, "actor2": ["B"] * 5})
        diversity = compute_actor_diversity(df)
        assert diversity > 0  # 2 actors

    def test_actor_diversity_many(self):
        df = pd.DataFrame(
            {
                "actor1": [f"Actor{i}" for i in range(20)],
                "actor2": [f"Group{i}" for i in range(20)],
            }
        )
        diversity = compute_actor_diversity(df)
        assert diversity > 0.8

    def test_civilian_targeting_ratio(self):
        df = pd.DataFrame(
            {
                "event_type": [
                    "Battles",
                    "Violence against civilians",
                    "Violence against civilians",
                    "Protests",
                ]
            }
        )
        ratio = compute_civilian_targeting_ratio(df)
        assert ratio == 0.5

    def test_rolling_event_rate(self, sample_events_df):
        result = compute_rolling_event_rate(sample_events_df)
        assert not result.empty
        assert "z_score" in result.columns


class TestSentimentFeatures:
    @pytest.fixture
    def tone_df(self):
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        tones = np.random.normal(-2, 3, 90)
        return pd.DataFrame(
            {
                "date": dates,
                "avg_tone": tones,
                "article_count": np.random.randint(10, 100, 90),
            }
        )

    def test_extract_sentiment_features(self, tone_df):
        features = extract_sentiment_features(tone_df, "Russia", "Ukraine")
        assert features is not None
        assert features.country_a == "Russia"
        assert features.country_b == "Ukraine"
        assert features.sentiment_category in (
            "hostile", "negative", "neutral", "positive", "cooperative"
        )
        assert isinstance(features.tone_z_score, float)

    def test_extract_insufficient_data(self):
        df = pd.DataFrame({"date": [datetime.utcnow()], "avg_tone": [0.0]})
        features = extract_sentiment_features(df, "A", "B")
        assert features is None

    def test_compute_sentiment_trajectory(self, tone_df):
        result = compute_sentiment_trajectory(tone_df)
        assert not result.empty
        assert "tone_ma_7d" in result.columns
        assert "momentum" in result.columns


class TestBaseRates:
    def test_base_rates_exist(self):
        assert len(HISTORICAL_BASE_RATES) > 0
        for key, rate in HISTORICAL_BASE_RATES.items():
            assert 0 <= rate.base_rate <= 1
            assert rate.sample_size > 0

    def test_get_base_rate(self):
        engine = BaseRateEngine()
        rate = engine.get_base_rate("military_buildup_to_invasion")
        assert rate is not None
        assert rate.base_rate == 0.15

    def test_regime_adjusted_rate(self):
        engine = BaseRateEngine()
        democracy_rate = engine.get_adjusted_rate(
            "military_buildup_to_invasion", regime_type="democracy"
        )
        autocracy_rate = engine.get_adjusted_rate(
            "military_buildup_to_invasion", regime_type="autocracy"
        )
        # Democracies less likely to invade
        assert democracy_rate < autocracy_rate

    def test_bayesian_update(self):
        engine = BaseRateEngine()
        # Evidence makes invasion more likely
        posterior = engine.compute_bayesian_update(
            "military_buildup_to_invasion",
            evidence_likelihood_ratio=5.0,
        )
        prior = 0.15
        assert posterior > prior

    def test_bayesian_update_negative_evidence(self):
        engine = BaseRateEngine()
        posterior = engine.compute_bayesian_update(
            "military_buildup_to_invasion",
            evidence_likelihood_ratio=0.2,
        )
        prior = 0.15
        assert posterior < prior


class TestEconomicFeatures:
    def test_interdependence_score(self):
        metrics = {
            "trade_as_pct_gdp_a": 5.0,
            "trade_as_pct_gdp_b": 10.0,
            "critical_supply_chains": ["oil", "gas", "metals"],
            "debt_holdings_a_of_b": 100_000_000_000,
            "debt_holdings_b_of_a": 50_000_000_000,
        }
        score = compute_interdependence_score(metrics)
        assert 0 <= score <= 1

    def test_escalation_constraint(self):
        low = compute_escalation_constraint(0.1)
        high = compute_escalation_constraint(0.9)
        assert high > low
        assert high <= 0.7  # Max constraint cap

    def test_economic_analyzer(self):
        analyzer = EconomicAnalyzer()
        # Use key format that matches BILATERAL_TRADE_DATA dict keys
        result = analyzer.get_bilateral_metrics("US", "China")
        assert result is not None
        assert result.bilateral_trade_volume > 0
        assert 0 <= result.interdependence_score <= 1
        assert 0 <= result.escalation_constraint <= 1


class TestPoliticalFeatures:
    def test_high_diversionary_risk(self):
        state = DomesticPoliticalState(
            country="TestCountry",
            leader_name="Test Leader",
            regime_type="autocracy",
            approval_rating=25.0,
            months_to_election=6,
            in_economic_recession=True,
            unemployment_rate=12.0,
            inflation_rate=15.0,
            ongoing_protests=True,
            protest_severity=0.8,
            recent_scandal=True,
            years_in_power=10,
            term_limited=False,
        )
        result = compute_diversionary_risk(state)
        assert result.diversionary_risk > 0.5
        assert result.risk_level in ("high", "very_high")
        assert len(result.risk_factors) > 0

    def test_low_diversionary_risk(self):
        state = DomesticPoliticalState(
            country="StableCountry",
            leader_name="Stable Leader",
            regime_type="democracy",
            approval_rating=65.0,
            months_to_election=36,
            in_economic_recession=False,
            unemployment_rate=4.0,
            inflation_rate=2.0,
            ongoing_protests=False,
            protest_severity=0.0,
            recent_scandal=False,
            years_in_power=2,
            term_limited=False,
        )
        result = compute_diversionary_risk(state)
        assert result.diversionary_risk < 0.3
        assert result.risk_level == "low"


class TestAllianceNetwork:
    def test_get_allies(self):
        analyzer = AllianceNetworkAnalyzer()
        allies = analyzer.get_allies("United States")
        assert len(allies) > 10  # US has many allies
        ally_names = [a["ally"] for a in allies]
        assert "United Kingdom" in ally_names
        assert "Japan" in ally_names

    def test_intervention_probability(self):
        analyzer = AllianceNetworkAnalyzer()
        analysis = analyzer.estimate_intervention_probability("Poland", "Russia")
        assert analysis.total_allies > 0
        assert analysis.defense_pact_allies > 0
        assert analysis.estimated_intervention_probability > 0.5  # NATO should intervene

    def test_no_allies(self):
        analyzer = AllianceNetworkAnalyzer()
        analysis = analyzer.estimate_intervention_probability("Andorra", "France")
        # Andorra has no modeled alliances
        assert analysis.estimated_intervention_probability <= 0.1

    def test_flash_points(self):
        analyzer = AllianceNetworkAnalyzer()
        flash_points = analyzer.find_flash_points()
        assert len(flash_points) > 0
        # US-China and US-Russia should appear
        pairs = [(f["country_a"], f["country_b"]) for f in flash_points]
        us_china = any(
            ("United States" in p and "China" in p) for p in pairs
        )
        assert us_china
