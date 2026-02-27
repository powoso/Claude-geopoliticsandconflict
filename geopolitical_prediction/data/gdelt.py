"""GDELT Project v2 API client for global news event data with tone/sentiment analysis.

GDELT monitors print, broadcast, and web news worldwide in 100+ languages,
updating every 15 minutes. We use it for:
  - Bilateral relationship sentiment tracking between country pairs
  - Crisis event detection and escalation monitoring
  - Media tone trajectory analysis for conflict/cooperation signals
"""

from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from geopolitical_prediction.config import GDELTConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class GDELTArticle:
    """A single GDELT document record."""

    url: str
    title: str
    source_country: str
    tone: float  # average tone (-100 to +100, negative = negative sentiment)
    positive_score: float
    negative_score: float
    polarity: float
    activity_ref_density: float
    self_group_density: float
    word_count: int
    date: datetime
    domain: str
    language: str
    themes: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)


@dataclass
class GDELTToneSeries:
    """Aggregated tone time-series for a query."""

    query: str
    timestamps: list[datetime] = field(default_factory=list)
    avg_tone: list[float] = field(default_factory=list)
    article_count: list[int] = field(default_factory=list)
    positive_pct: list[float] = field(default_factory=list)
    negative_pct: list[float] = field(default_factory=list)


class GDELTClient:
    """Client for GDELT v2 DOC, GEO, and TV APIs."""

    def __init__(self, config: GDELTConfig | None = None):
        self.config = config or get_settings().gdelt
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GeopoliticalPredictionMarkets/0.1"})
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str, params: dict[str, Any]) -> requests.Response:
        self._rate_limit()
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp

    # ----- DOC API: Article search with tone analysis -----

    def search_articles(
        self,
        query: str,
        mode: str = "ArtList",
        max_records: int = 250,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source_country: str | None = None,
        source_lang: str = "english",
        sort: str = "DateDesc",
    ) -> list[GDELTArticle]:
        """Search GDELT DOC API for articles matching a query.

        Args:
            query: Search terms (e.g., "Russia Ukraine invasion").
            mode: ArtList (articles) or TimelineVol/TimelineTone (timeseries).
            max_records: Max articles to return (max 250 per request).
            start_date: Filter start (GDELT keeps ~3 months online).
            end_date: Filter end.
            source_country: ISO 2-letter country code for source filtering.
            source_lang: Language filter.
            sort: DateDesc, DateAsc, ToneDesc, ToneAsc, HybridRel.
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "maxrecords": min(max_records, 250),
            "format": "json",
            "sort": sort,
        }
        if start_date:
            params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
        if end_date:
            params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")
        if source_country:
            params["sourcecountry"] = source_country
        if source_lang:
            params["sourcelang"] = source_lang

        try:
            resp = self._request(self.config.doc_api_url, params)
            data = resp.json()
        except Exception:
            logger.exception("GDELT DOC API request failed for query=%s", query)
            return []

        articles = []
        for item in data.get("articles", []):
            try:
                tone_parts = str(item.get("tone", "0,0,0,0,0,0")).split(",")
                tone_floats = [float(t) for t in tone_parts] + [0.0] * 6
                articles.append(
                    GDELTArticle(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        source_country=item.get("sourcecountry", ""),
                        tone=tone_floats[0],
                        positive_score=tone_floats[1],
                        negative_score=tone_floats[2],
                        polarity=tone_floats[3],
                        activity_ref_density=tone_floats[4],
                        self_group_density=tone_floats[5],
                        word_count=int(item.get("wordcount", 0)),
                        date=datetime.strptime(
                            item.get("seendate", "19700101T000000Z"),
                            "%Y%m%dT%H%M%SZ",
                        ),
                        domain=item.get("domain", ""),
                        language=item.get("language", ""),
                        themes=item.get("themes", "").split(";")
                        if item.get("themes")
                        else [],
                        locations=item.get("locations", "").split(";")
                        if item.get("locations")
                        else [],
                        persons=item.get("persons", "").split(";")
                        if item.get("persons")
                        else [],
                        organizations=item.get("organizations", "").split(";")
                        if item.get("organizations")
                        else [],
                    )
                )
            except (ValueError, KeyError):
                logger.warning("Skipping malformed GDELT article: %s", item.get("url"))
                continue
        logger.info("GDELT returned %d articles for query='%s'", len(articles), query)
        return articles

    def get_tone_timeline(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source_lang: str = "english",
        resolution: str = "day",
    ) -> GDELTToneSeries:
        """Get average tone over time for a query (TimelineTone mode).

        Returns a time series of average tone and volume, useful for
        tracking sentiment trajectories in bilateral relationships.
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": "TimelineTone",
            "format": "json",
            "sourcelang": source_lang,
            "TIMELINESMOOTH": 0 if resolution == "day" else 5,
        }
        if start_date:
            params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
        if end_date:
            params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")

        try:
            resp = self._request(self.config.doc_api_url, params)
            data = resp.json()
        except Exception:
            logger.exception("GDELT tone timeline request failed for query=%s", query)
            return GDELTToneSeries(query=query)

        series = GDELTToneSeries(query=query)
        timeline = data.get("timeline", [])
        if timeline and isinstance(timeline, list):
            # GDELT returns [{series, data: [{date, value}]}]
            for entry in timeline:
                for point in entry.get("data", []):
                    try:
                        series.timestamps.append(
                            datetime.strptime(point["date"], "%Y-%m-%dT%H:%M:%SZ")
                        )
                        series.avg_tone.append(float(point["value"]))
                    except (ValueError, KeyError):
                        continue
        logger.info(
            "GDELT tone timeline: %d points for query='%s'",
            len(series.timestamps),
            query,
        )
        return series

    def get_volume_timeline(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source_lang: str = "english",
    ) -> GDELTToneSeries:
        """Get article volume over time (TimelineVol mode).

        Useful for detecting media attention spikes during crises.
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": "TimelineVol",
            "format": "json",
            "sourcelang": source_lang,
        }
        if start_date:
            params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
        if end_date:
            params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")

        try:
            resp = self._request(self.config.doc_api_url, params)
            data = resp.json()
        except Exception:
            logger.exception("GDELT volume timeline request failed for query=%s", query)
            return GDELTToneSeries(query=query)

        series = GDELTToneSeries(query=query)
        timeline = data.get("timeline", [])
        if timeline and isinstance(timeline, list):
            for entry in timeline:
                for point in entry.get("data", []):
                    try:
                        series.timestamps.append(
                            datetime.strptime(point["date"], "%Y-%m-%dT%H:%M:%SZ")
                        )
                        series.article_count.append(int(float(point["value"])))
                    except (ValueError, KeyError):
                        continue
        return series

    # ----- GEO API: Geospatial event mapping -----

    def geo_search(
        self,
        query: str,
        mode: str = "PointData",
        max_points: int = 1000,
        source_lang: str = "english",
    ) -> pd.DataFrame:
        """Search GDELT GEO API for geolocated event mentions.

        Returns a DataFrame with lat/lon, tone, article count per location.
        Useful for mapping where geopolitical events are concentrated.
        """
        params: dict[str, Any] = {
            "query": query,
            "format": "GeoJSON",
            "mode": mode,
            "sourcelang": source_lang,
            "maxpoints": max_points,
        }

        try:
            resp = self._request(self.config.geo_api_url, params)
            data = resp.json()
        except Exception:
            logger.exception("GDELT GEO API request failed for query=%s", query)
            return pd.DataFrame()

        rows = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [0, 0])
            rows.append(
                {
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "name": props.get("name", ""),
                    "article_count": props.get("count", 0),
                    "avg_tone": props.get("tone", 0.0),
                    "url": props.get("url", ""),
                }
            )
        return pd.DataFrame(rows)

    # ----- Bilateral relationship sentiment -----

    def bilateral_sentiment(
        self,
        country_a: str,
        country_b: str,
        days_back: int = 90,
    ) -> pd.DataFrame:
        """Track media sentiment between two countries over time.

        Constructs query like "Russia Ukraine" and returns daily tone.
        This is a key input to the escalation ladder model.
        """
        query = f'"{country_a}" "{country_b}"'
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)

        tone_series = self.get_tone_timeline(query, start_date=start, end_date=end)
        vol_series = self.get_volume_timeline(query, start_date=start, end_date=end)

        if not tone_series.timestamps:
            return pd.DataFrame(
                columns=["date", "avg_tone", "article_count", "country_a", "country_b"]
            )

        df = pd.DataFrame(
            {
                "date": tone_series.timestamps,
                "avg_tone": tone_series.avg_tone,
            }
        )
        df["country_a"] = country_a
        df["country_b"] = country_b

        if vol_series.timestamps:
            vol_df = pd.DataFrame(
                {
                    "date": vol_series.timestamps,
                    "article_count": vol_series.article_count,
                }
            )
            df = df.merge(vol_df, on="date", how="left")
        else:
            df["article_count"] = 0

        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ----- GKG (Global Knowledge Graph) latest update -----

    def get_latest_gkg_url(self) -> str | None:
        """Fetch the URL of the latest GKG update file.

        GDELT publishes a new GKG file every 15 minutes. This returns the
        URL for the most recent one, which can be downloaded and parsed for
        themes, persons, organizations, and tone.
        """
        try:
            resp = self._request(self.config.gkg_url, {})
            lines = resp.text.strip().split("\n")
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3 and "gkg" in parts[2].lower():
                    return parts[2]
        except Exception:
            logger.exception("Failed to fetch latest GKG URL")
        return None

    def download_gkg_update(self) -> pd.DataFrame:
        """Download and parse the latest 15-minute GKG update.

        Returns a DataFrame with columns for themes, persons, organizations,
        locations, tone, and source URLs.
        """
        url = self.get_latest_gkg_url()
        if not url:
            return pd.DataFrame()

        try:
            import zipfile

            self._rate_limit()
            resp = self.session.get(url, timeout=120)
            resp.raise_for_status()

            zip_buffer = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_buffer) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    text = f.read().decode("utf-8", errors="replace")

            # GKG v2 has many columns; we extract the most useful ones
            gkg_cols = [
                "gkg_record_id",
                "date",
                "source_collection_id",
                "source_common_name",
                "document_id",
                "counts",
                "v2_counts",
                "themes",
                "v2_themes",
                "locations",
                "v2_locations",
                "persons",
                "v2_persons",
                "organizations",
                "v2_organizations",
                "v2_tone",
                "dates",
                "gcam",
                "sharing_image",
                "related_images",
                "social_image_embeds",
                "social_video_embeds",
                "quotations",
                "all_names",
                "amounts",
                "translation_info",
                "extras",
            ]
            reader = csv.reader(io.StringIO(text), delimiter="\t")
            rows = []
            for row in reader:
                if len(row) >= 16:
                    tone_parts = row[15].split(",") if row[15] else ["0"] * 7
                    tone_vals = [float(t) for t in tone_parts[:7]] + [0.0] * 7
                    rows.append(
                        {
                            "record_id": row[0],
                            "date": row[1],
                            "source": row[3],
                            "themes": row[7] if len(row) > 7 else "",
                            "locations": row[9] if len(row) > 9 else "",
                            "persons": row[11] if len(row) > 11 else "",
                            "organizations": row[13] if len(row) > 13 else "",
                            "avg_tone": tone_vals[0],
                            "positive_score": tone_vals[1],
                            "negative_score": tone_vals[2],
                            "polarity": tone_vals[3],
                        }
                    )
            logger.info("Parsed %d GKG records from latest update", len(rows))
            return pd.DataFrame(rows)
        except Exception:
            logger.exception("Failed to download/parse GKG update")
            return pd.DataFrame()


class GDELTSentimentPipeline:
    """High-level pipeline for continuous GDELT sentiment monitoring.

    Wraps the GDELTClient to provide a streaming-style interface for
    tracking multiple country pairs and crisis keywords.
    """

    def __init__(self, config: GDELTConfig | None = None):
        self.client = GDELTClient(config)
        self.watched_pairs: list[tuple[str, str]] = []
        self.watched_keywords: list[str] = []
        self._history: dict[str, pd.DataFrame] = {}

    def watch_country_pair(self, country_a: str, country_b: str) -> None:
        """Add a country pair to monitor for bilateral sentiment."""
        pair = (country_a, country_b)
        if pair not in self.watched_pairs:
            self.watched_pairs.append(pair)
            logger.info("Now watching bilateral sentiment: %s <-> %s", country_a, country_b)

    def watch_keyword(self, keyword: str) -> None:
        """Add a crisis keyword to monitor."""
        if keyword not in self.watched_keywords:
            self.watched_keywords.append(keyword)

    def update_all(self, days_back: int = 30) -> dict[str, pd.DataFrame]:
        """Refresh sentiment data for all watched pairs and keywords.

        Returns dict mapping pair/keyword identifiers to DataFrames.
        """
        results: dict[str, pd.DataFrame] = {}

        for country_a, country_b in self.watched_pairs:
            key = f"{country_a}_{country_b}"
            try:
                df = self.client.bilateral_sentiment(country_a, country_b, days_back)
                results[key] = df
                self._history[key] = df
                logger.info(
                    "Updated bilateral sentiment %s: %d records", key, len(df)
                )
            except Exception:
                logger.exception("Failed to update bilateral sentiment for %s", key)

        for keyword in self.watched_keywords:
            try:
                end = datetime.utcnow()
                start = end - timedelta(days=days_back)
                tone = self.client.get_tone_timeline(keyword, start, end)
                df = pd.DataFrame(
                    {
                        "date": tone.timestamps,
                        "avg_tone": tone.avg_tone,
                        "keyword": keyword,
                    }
                )
                results[f"kw_{keyword}"] = df
                self._history[f"kw_{keyword}"] = df
            except Exception:
                logger.exception("Failed to update keyword sentiment for %s", keyword)

        return results

    def get_latest_sentiment(self, key: str) -> float | None:
        """Get the most recent tone value for a watched pair/keyword."""
        df = self._history.get(key)
        if df is not None and not df.empty and "avg_tone" in df.columns:
            return float(df["avg_tone"].iloc[-1])
        return None

    def detect_sentiment_shift(
        self,
        key: str,
        short_window: int = 7,
        long_window: int = 30,
        threshold: float = 2.0,
    ) -> dict[str, Any] | None:
        """Detect significant sentiment shifts for a monitored entity.

        Compares short-term moving average to long-term baseline.
        Returns shift details if the divergence exceeds the threshold.
        """
        df = self._history.get(key)
        if df is None or len(df) < long_window:
            return None

        recent = df["avg_tone"].tail(short_window).mean()
        baseline = df["avg_tone"].tail(long_window).mean()
        std = df["avg_tone"].tail(long_window).std()

        if std == 0:
            return None

        z_score = (recent - baseline) / std

        if abs(z_score) >= threshold:
            return {
                "key": key,
                "recent_avg_tone": recent,
                "baseline_avg_tone": baseline,
                "z_score": z_score,
                "direction": "deteriorating" if z_score < 0 else "improving",
                "timestamp": datetime.utcnow().isoformat(),
            }
        return None

    def detect_all_shifts(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Check all monitored entities for sentiment shifts."""
        shifts = []
        for key in self._history:
            shift = self.detect_sentiment_shift(key, **kwargs)
            if shift:
                shifts.append(shift)
        return shifts
