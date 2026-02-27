"""Tests for ACLED data collection and conflict pipeline."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from geopolitical_prediction.config import ACLEDConfig
from geopolitical_prediction.data.acled import (
    ACLEDClient,
    ACLEDConflictPipeline,
    ACLEDEvent,
    EventType,
    EVENT_SEVERITY_WEIGHTS,
)


@pytest.fixture
def acled_config():
    return ACLEDConfig(api_key="test_key", email="test@example.com")


@pytest.fixture
def client(acled_config):
    return ACLEDClient(acled_config)


@pytest.fixture
def mock_acled_response():
    return {
        "success": True,
        "data": [
            {
                "event_id_cnty": "UKR12345",
                "event_date": "2024-01-15",
                "event_type": "Battles",
                "sub_event_type": "Armed clash",
                "actor1": "Military Forces of Ukraine",
                "actor2": "Military Forces of Russia",
                "country": "Ukraine",
                "iso3": "UKR",
                "admin1": "Donetsk",
                "admin2": "Bakhmut",
                "admin3": "",
                "location": "Bakhmut",
                "latitude": "48.5953",
                "longitude": "38.0009",
                "fatalities": "5",
                "notes": "Clashes reported near Bakhmut",
                "source": "Ukrainian MoD",
                "interaction": "12",
                "timestamp": "2024-01-15",
            },
            {
                "event_id_cnty": "UKR12346",
                "event_date": "2024-01-15",
                "event_type": "Explosions/Remote violence",
                "sub_event_type": "Shelling/artillery/missile attack",
                "actor1": "Military Forces of Russia",
                "actor2": "",
                "country": "Ukraine",
                "iso3": "UKR",
                "admin1": "Kherson",
                "admin2": "",
                "admin3": "",
                "location": "Kherson",
                "latitude": "46.6354",
                "longitude": "32.6169",
                "fatalities": "2",
                "notes": "Shelling of Kherson city",
                "source": "Kherson OVA",
                "interaction": "10",
                "timestamp": "2024-01-15",
            },
        ],
    }


class TestACLEDClient:
    def test_fetch_events(self, client, mock_acled_response):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_acled_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            events = client.fetch_events(country="Ukraine")

            assert len(events) == 2
            assert events[0].event_id == "UKR12345"
            assert events[0].event_type == "Battles"
            assert events[0].fatalities == 5
            assert events[0].latitude == 48.5953
            assert events[1].event_type == "Explosions/Remote violence"

    def test_event_severity_weights(self):
        event = ACLEDEvent(
            event_id="test",
            event_date=datetime.utcnow(),
            event_type="Battles",
            sub_event_type="Armed clash",
            actor1="A",
            actor2="B",
            country="Test",
            iso3="TST",
            admin1="",
            admin2="",
            admin3="",
            location="TestCity",
            latitude=0.0,
            longitude=0.0,
            fatalities=10,
            notes="",
            source="",
            interaction=12,
        )

        assert event.severity_weight == 0.8
        assert event.weighted_severity > event.severity_weight  # fatalities boost it

    def test_weighted_severity_capped(self):
        """High fatalities should be capped at 5x multiplier."""
        event = ACLEDEvent(
            event_id="test",
            event_date=datetime.utcnow(),
            event_type="Battles",
            sub_event_type="Armed clash",
            actor1="A",
            actor2="B",
            country="Test",
            iso3="TST",
            admin1="",
            admin2="",
            admin3="",
            location="TestCity",
            latitude=0.0,
            longitude=0.0,
            fatalities=10000,
            notes="",
            source="",
            interaction=12,
        )
        # Max multiplier is 5.0 regardless of fatalities
        assert event.weighted_severity == event.severity_weight * 5.0

    def test_fetch_events_handles_api_error(self, client):
        with patch.object(client.session, "get", side_effect=Exception("Timeout")):
            events = client.fetch_events(country="Ukraine")
            assert events == []

    def test_fetch_events_df(self, client, mock_acled_response):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_acled_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            df = client.fetch_events_df(country="Ukraine")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "weighted_severity" in df.columns

    def test_date_range_param(self):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)
        result = ACLEDClient._date_range_param(start, end)
        assert result == "2024-01-01|2024-06-30"

    def test_date_range_param_none(self):
        assert ACLEDClient._date_range_param(None, None) is None


class TestACLEDConflictPipeline:
    def test_watch_country(self):
        pipeline = ACLEDConflictPipeline(ACLEDConfig(api_key="test", email="test@test.com"))
        pipeline.watch_country("Ukraine", "UKR")
        pipeline.watch_country("Sudan", "SDN")

        assert len(pipeline.watched_countries) == 2

    def test_no_duplicate_watch(self):
        pipeline = ACLEDConflictPipeline(ACLEDConfig(api_key="test", email="test@test.com"))
        pipeline.watch_country("Ukraine", "UKR")
        pipeline.watch_country("Ukraine", "UKR")

        assert len(pipeline.watched_countries) == 1

    def test_compute_snapshot(self):
        pipeline = ACLEDConflictPipeline(ACLEDConfig(api_key="test", email="test@test.com"))

        # Create synthetic event data
        now = datetime.utcnow()
        events = []
        for i in range(50):
            events.append(
                {
                    "event_id": f"EVT{i}",
                    "event_date": now - timedelta(days=i % 30),
                    "event_type": "Battles" if i % 3 == 0 else "Explosions/Remote violence",
                    "sub_event_type": "Armed clash",
                    "actor1": "Force A",
                    "actor2": "Force B",
                    "country": "TestCountry",
                    "iso3": "TST",
                    "admin1": "Region",
                    "location": f"Location{i % 5}",
                    "latitude": 48.0 + (i % 10) * 0.1,
                    "longitude": 38.0 + (i % 10) * 0.1,
                    "fatalities": i % 5,
                    "severity_weight": 0.8,
                    "weighted_severity": 0.8 * (1.0 + (i % 5) / 100.0),
                    "interaction": 12,
                    "notes": "",
                }
            )

        pipeline._event_cache["TST"] = pd.DataFrame(events)
        pipeline._event_cache["TST"]["event_date"] = pd.to_datetime(
            pipeline._event_cache["TST"]["event_date"]
        )

        snapshot = pipeline.compute_snapshot("TST", window_days=30)
        assert snapshot is not None
        assert snapshot.country == "TestCountry"
        assert snapshot.total_events > 0
        assert 0 <= snapshot.escalation_score <= 1
        assert len(snapshot.hotspot_locations) > 0

    def test_compute_escalation_trend(self):
        pipeline = ACLEDConflictPipeline(ACLEDConfig(api_key="test", email="test@test.com"))

        now = datetime.utcnow()
        events = []
        # Create strongly escalating pattern:
        # last 7 days: 10 events/day with high fatalities
        # days 8-30: 1 event/day with low fatalities
        for i in range(30):
            days_ago = i
            if days_ago < 7:
                n_events = 10
                fatalities = 20
            else:
                n_events = 1
                fatalities = 1
            for j in range(n_events):
                events.append(
                    {
                        "event_id": f"EVT{i}_{j}",
                        "event_date": now - timedelta(days=days_ago),
                        "event_type": "Battles",
                        "sub_event_type": "Armed clash",
                        "actor1": "A",
                        "actor2": "B",
                        "country": "TestCountry",
                        "iso3": "TST",
                        "admin1": "",
                        "location": "Location",
                        "latitude": 48.0,
                        "longitude": 38.0,
                        "fatalities": fatalities,
                        "severity_weight": 0.8,
                        "weighted_severity": 0.8 * (1 + fatalities / 100),
                        "interaction": 12,
                        "notes": "",
                    }
                )

        pipeline._event_cache["TST"] = pd.DataFrame(events)
        pipeline._event_cache["TST"]["event_date"] = pd.to_datetime(
            pipeline._event_cache["TST"]["event_date"]
        )

        trend = pipeline.compute_escalation_trend("TST", short_window=7, long_window=30)
        assert trend is not None
        assert trend["trend"] in (
            "escalating_rapidly",
            "escalating",
            "stable",
            "de_escalating",
            "de_escalating_rapidly",
        )
        # With 10x more events and much higher fatalities recently
        assert trend["composite_ratio"] > 1.0
