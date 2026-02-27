"""Tests for GDELT data collection and sentiment pipeline."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from geopolitical_prediction.config import GDELTConfig
from geopolitical_prediction.data.gdelt import (
    GDELTArticle,
    GDELTClient,
    GDELTSentimentPipeline,
    GDELTToneSeries,
)


@pytest.fixture
def gdelt_config():
    return GDELTConfig(request_delay=0.0)


@pytest.fixture
def client(gdelt_config):
    return GDELTClient(gdelt_config)


@pytest.fixture
def mock_article_response():
    return {
        "articles": [
            {
                "url": "https://example.com/article1",
                "title": "Russia-Ukraine tensions escalate",
                "sourcecountry": "US",
                "tone": "-3.5,1.2,4.7,5.9,0.5,0.3",
                "wordcount": 500,
                "seendate": "20240115T120000Z",
                "domain": "example.com",
                "language": "English",
                "themes": "MILITARY;CONFLICT",
                "locations": "Russia;Ukraine",
                "persons": "",
                "organizations": "NATO",
            },
            {
                "url": "https://example.com/article2",
                "title": "Peace talks resume",
                "sourcecountry": "UK",
                "tone": "2.1,3.0,0.9,3.9,0.2,0.1",
                "wordcount": 350,
                "seendate": "20240115T140000Z",
                "domain": "example.com",
                "language": "English",
                "themes": "DIPLOMACY",
                "locations": "Geneva",
                "persons": "",
                "organizations": "UN",
            },
        ]
    }


@pytest.fixture
def mock_tone_timeline_response():
    return {
        "timeline": [
            {
                "series": "Overall Tone",
                "data": [
                    {"date": "2024-01-10T00:00:00Z", "value": -2.5},
                    {"date": "2024-01-11T00:00:00Z", "value": -3.1},
                    {"date": "2024-01-12T00:00:00Z", "value": -1.8},
                    {"date": "2024-01-13T00:00:00Z", "value": -4.2},
                    {"date": "2024-01-14T00:00:00Z", "value": -3.9},
                ],
            }
        ]
    }


class TestGDELTClient:
    def test_search_articles(self, client, mock_article_response):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_article_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            articles = client.search_articles("Russia Ukraine")

            assert len(articles) == 2
            assert articles[0].title == "Russia-Ukraine tensions escalate"
            assert articles[0].tone == -3.5
            assert articles[0].positive_score == 1.2
            assert articles[0].word_count == 500
            assert "MILITARY" in articles[0].themes
            assert articles[1].tone == 2.1

    def test_get_tone_timeline(self, client, mock_tone_timeline_response):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_tone_timeline_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            series = client.get_tone_timeline("Russia Ukraine")

            assert series.query == "Russia Ukraine"
            assert len(series.timestamps) == 5
            assert series.avg_tone[0] == -2.5
            assert series.avg_tone[3] == -4.2

    def test_search_articles_handles_api_error(self, client):
        with patch.object(client.session, "get", side_effect=Exception("Network error")):
            articles = client.search_articles("test query")
            assert articles == []

    def test_search_articles_handles_malformed_data(self, client):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "articles": [
                    {"url": "bad", "tone": "not_a_number"},
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            articles = client.search_articles("test")
            # Should skip malformed articles gracefully
            assert len(articles) == 0

    def test_bilateral_sentiment(self, client, mock_tone_timeline_response):
        with patch.object(client.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_tone_timeline_response
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            df = client.bilateral_sentiment("Russia", "Ukraine", days_back=30)
            assert isinstance(df, pd.DataFrame)
            assert "avg_tone" in df.columns
            assert "country_a" in df.columns


class TestGDELTSentimentPipeline:
    def test_watch_country_pair(self):
        pipeline = GDELTSentimentPipeline(GDELTConfig(request_delay=0.0))
        pipeline.watch_country_pair("Russia", "Ukraine")
        pipeline.watch_country_pair("China", "Taiwan")

        assert len(pipeline.watched_pairs) == 2
        assert ("Russia", "Ukraine") in pipeline.watched_pairs

    def test_watch_keyword(self):
        pipeline = GDELTSentimentPipeline(GDELTConfig(request_delay=0.0))
        pipeline.watch_keyword("sanctions")
        pipeline.watch_keyword("invasion")

        assert len(pipeline.watched_keywords) == 2

    def test_no_duplicate_watches(self):
        pipeline = GDELTSentimentPipeline(GDELTConfig(request_delay=0.0))
        pipeline.watch_country_pair("Russia", "Ukraine")
        pipeline.watch_country_pair("Russia", "Ukraine")

        assert len(pipeline.watched_pairs) == 1

    def test_detect_sentiment_shift(self):
        pipeline = GDELTSentimentPipeline(GDELTConfig(request_delay=0.0))

        # Create synthetic history with a clear shift
        # 90 days of normal noise, then 5 days of sharp drop
        # Use long_window=90 so the drop is a clear anomaly vs baseline
        import numpy as np
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=95, freq="D")
        baseline = np.random.normal(0.0, 1.0, 90).tolist()
        drop = [-15.0] * 5
        tones = baseline + drop
        pipeline._history["Russia_Ukraine"] = pd.DataFrame(
            {"date": dates, "avg_tone": tones}
        )

        shift = pipeline.detect_sentiment_shift(
            "Russia_Ukraine", short_window=5, long_window=90, threshold=2.0
        )
        assert shift is not None
        assert shift["direction"] == "deteriorating"
        assert shift["z_score"] < -2.0

    def test_no_shift_when_stable(self):
        pipeline = GDELTSentimentPipeline(GDELTConfig(request_delay=0.0))

        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        tones = [0.0] * 60
        pipeline._history["stable_pair"] = pd.DataFrame(
            {"date": dates, "avg_tone": tones}
        )

        shift = pipeline.detect_sentiment_shift("stable_pair")
        assert shift is None
