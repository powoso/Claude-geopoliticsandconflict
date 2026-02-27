"""ACLED (Armed Conflict Location & Event Data) API client.

ACLED tracks political violence and protest events worldwide with precise
geolocation and event classification. We use it for:
  - Conflict escalation scoring (frequency/severity over rolling windows)
  - Geographic hotspot detection
  - Event type distribution analysis (battles vs protests vs violence against civilians)
  - Fatality trend monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
import requests

from geopolitical_prediction.config import ACLEDConfig, get_settings

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """ACLED event type classification."""

    BATTLES = "Battles"
    VIOLENCE_AGAINST_CIVILIANS = "Violence against civilians"
    EXPLOSIONS_REMOTE_VIOLENCE = "Explosions/Remote violence"
    RIOTS = "Riots"
    PROTESTS = "Protests"
    STRATEGIC_DEVELOPMENTS = "Strategic developments"


class SubEventType(str, Enum):
    """Selected high-signal sub-event types for escalation detection."""

    ARMED_CLASH = "Armed clash"
    GOVERNMENT_REGAINS_TERRITORY = "Government regains territory"
    NON_STATE_ACTOR_OVERTAKES_TERRITORY = "Non-state actor overtakes territory"
    AIR_DRONE_STRIKE = "Air/drone strike"
    SHELLING_ARTILLERY = "Shelling/artillery/missile attack"
    SUICIDE_BOMB = "Suicide bomb"
    CHEMICAL_WEAPON = "Chemical weapon"
    ABDUCTION_FORCED_DISAPPEARANCE = "Abduction/forced disappearance"
    SEXUAL_VIOLENCE = "Sexual violence"
    EXCESSIVE_FORCE = "Excessive force against protesters"
    PEACEFUL_PROTEST = "Peaceful protest"
    MOB_VIOLENCE = "Mob violence"


# Severity weights for escalation scoring
EVENT_SEVERITY_WEIGHTS: dict[str, float] = {
    EventType.BATTLES: 0.8,
    EventType.VIOLENCE_AGAINST_CIVILIANS: 0.9,
    EventType.EXPLOSIONS_REMOTE_VIOLENCE: 0.85,
    EventType.RIOTS: 0.5,
    EventType.PROTESTS: 0.2,
    EventType.STRATEGIC_DEVELOPMENTS: 0.4,
}


@dataclass
class ACLEDEvent:
    """A single ACLED conflict event record."""

    event_id: str
    event_date: datetime
    event_type: str
    sub_event_type: str
    actor1: str
    actor2: str
    country: str
    iso3: str
    admin1: str  # first-level administrative division
    admin2: str
    admin3: str
    location: str
    latitude: float
    longitude: float
    fatalities: int
    notes: str
    source: str
    interaction: int  # ACLED interaction code (who fights whom)
    timestamp: datetime | None = None

    @property
    def severity_weight(self) -> float:
        return EVENT_SEVERITY_WEIGHTS.get(self.event_type, 0.3)

    @property
    def weighted_severity(self) -> float:
        """Combine event type weight with fatality count for scoring."""
        fatality_factor = min(1.0 + (self.fatalities / 100.0), 5.0)
        return self.severity_weight * fatality_factor


@dataclass
class ConflictSnapshot:
    """Aggregated conflict statistics for a country over a time window."""

    country: str
    iso3: str
    window_start: datetime
    window_end: datetime
    total_events: int = 0
    total_fatalities: int = 0
    battles: int = 0
    explosions: int = 0
    violence_against_civilians: int = 0
    riots: int = 0
    protests: int = 0
    strategic_developments: int = 0
    avg_severity: float = 0.0
    escalation_score: float = 0.0
    hotspot_locations: list[dict[str, Any]] = field(default_factory=list)


class ACLEDClient:
    """Client for the ACLED REST API."""

    def __init__(self, config: ACLEDConfig | None = None):
        self.config = config or get_settings().acled
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GeopoliticalPredictionMarkets/0.1"})

    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "key": self.config.api_key,
            "email": self.config.email,
            "limit": self.config.page_size,
        }
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def fetch_events(
        self,
        country: str | None = None,
        iso3: str | None = None,
        event_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        admin1: str | None = None,
        limit: int | None = None,
        page: int = 1,
    ) -> list[ACLEDEvent]:
        """Fetch conflict events from ACLED with optional filters.

        Args:
            country: Country name filter.
            iso3: ISO 3166-1 alpha-3 country code.
            event_type: Filter by ACLED event type.
            start_date: Events on or after this date.
            end_date: Events on or before this date.
            admin1: First-level administrative region.
            limit: Max events to return.
            page: Page number for pagination.
        """
        params = self._build_params(
            country=country,
            iso=iso3,
            event_type=event_type,
            event_date=self._date_range_param(start_date, end_date),
            admin1=admin1,
            limit=limit or self.config.page_size,
            page=page,
        )

        try:
            resp = self.session.get(self.config.base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("ACLED API request failed")
            return []

        if not data.get("success", True):
            logger.error("ACLED API error: %s", data.get("error", "unknown"))
            return []

        events = []
        for row in data.get("data", []):
            try:
                events.append(self._parse_event(row))
            except (ValueError, KeyError):
                logger.warning("Skipping malformed ACLED event: %s", row.get("event_id_cnty"))
                continue

        logger.info(
            "ACLED returned %d events (country=%s, page=%d)", len(events), country, page
        )
        return events

    def fetch_all_events(
        self,
        country: str | None = None,
        iso3: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_pages: int = 20,
    ) -> list[ACLEDEvent]:
        """Fetch all events with automatic pagination."""
        all_events: list[ACLEDEvent] = []
        for page in range(1, max_pages + 1):
            events = self.fetch_events(
                country=country,
                iso3=iso3,
                start_date=start_date,
                end_date=end_date,
                page=page,
            )
            all_events.extend(events)
            if len(events) < self.config.page_size:
                break
        logger.info("ACLED total fetched: %d events across %d pages", len(all_events), page)
        return all_events

    def fetch_events_df(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch events and return as a DataFrame."""
        events = self.fetch_all_events(**kwargs)
        if not events:
            return pd.DataFrame()

        records = []
        for e in events:
            records.append(
                {
                    "event_id": e.event_id,
                    "event_date": e.event_date,
                    "event_type": e.event_type,
                    "sub_event_type": e.sub_event_type,
                    "actor1": e.actor1,
                    "actor2": e.actor2,
                    "country": e.country,
                    "iso3": e.iso3,
                    "admin1": e.admin1,
                    "location": e.location,
                    "latitude": e.latitude,
                    "longitude": e.longitude,
                    "fatalities": e.fatalities,
                    "severity_weight": e.severity_weight,
                    "weighted_severity": e.weighted_severity,
                    "interaction": e.interaction,
                    "notes": e.notes,
                }
            )
        df = pd.DataFrame(records)
        df["event_date"] = pd.to_datetime(df["event_date"])
        return df.sort_values("event_date").reset_index(drop=True)

    @staticmethod
    def _date_range_param(
        start: datetime | None, end: datetime | None
    ) -> str | None:
        if start and end:
            return f"{start.strftime('%Y-%m-%d')}|{end.strftime('%Y-%m-%d')}"
        elif start:
            return f"{start.strftime('%Y-%m-%d')}|{datetime.utcnow().strftime('%Y-%m-%d')}"
        elif end:
            return f"1997-01-01|{end.strftime('%Y-%m-%d')}"
        return None

    @staticmethod
    def _parse_event(row: dict[str, Any]) -> ACLEDEvent:
        return ACLEDEvent(
            event_id=str(row.get("event_id_cnty", "")),
            event_date=datetime.strptime(row["event_date"], "%Y-%m-%d"),
            event_type=row.get("event_type", ""),
            sub_event_type=row.get("sub_event_type", ""),
            actor1=row.get("actor1", ""),
            actor2=row.get("actor2", ""),
            country=row.get("country", ""),
            iso3=row.get("iso3", ""),
            admin1=row.get("admin1", ""),
            admin2=row.get("admin2", ""),
            admin3=row.get("admin3", ""),
            location=row.get("location", ""),
            latitude=float(row.get("latitude", 0)),
            longitude=float(row.get("longitude", 0)),
            fatalities=int(row.get("fatalities", 0)),
            notes=row.get("notes", ""),
            source=row.get("source", ""),
            interaction=int(row.get("interaction", 0)),
            timestamp=datetime.strptime(row["timestamp"], "%Y-%m-%d")
            if row.get("timestamp")
            else None,
        )


class ACLEDConflictPipeline:
    """High-level pipeline for continuous conflict event monitoring.

    Wraps ACLEDClient to provide rolling-window analysis, escalation
    scoring, and hotspot detection for multiple countries.
    """

    def __init__(self, config: ACLEDConfig | None = None):
        self.client = ACLEDClient(config)
        self.watched_countries: list[dict[str, str]] = []
        self._event_cache: dict[str, pd.DataFrame] = {}

    def watch_country(self, country: str, iso3: str = "") -> None:
        """Add a country to the monitoring list."""
        entry = {"country": country, "iso3": iso3}
        if entry not in self.watched_countries:
            self.watched_countries.append(entry)
            logger.info("Now watching conflict events for: %s", country)

    def update_all(self, days_back: int = 90) -> dict[str, pd.DataFrame]:
        """Refresh event data for all watched countries."""
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)
        results: dict[str, pd.DataFrame] = {}

        for entry in self.watched_countries:
            key = entry["iso3"] or entry["country"]
            try:
                df = self.client.fetch_events_df(
                    country=entry["country"],
                    iso3=entry["iso3"] or None,
                    start_date=start,
                    end_date=end,
                )
                results[key] = df
                self._event_cache[key] = df
                logger.info("Updated ACLED data for %s: %d events", key, len(df))
            except Exception:
                logger.exception("Failed to update ACLED data for %s", key)
        return results

    def compute_snapshot(
        self,
        country_key: str,
        window_days: int = 30,
    ) -> ConflictSnapshot | None:
        """Compute aggregated conflict snapshot for a country.

        Calculates event counts by type, fatalities, severity scores,
        and geographic hotspots over a rolling window.
        """
        df = self._event_cache.get(country_key)
        if df is None or df.empty:
            return None

        cutoff = datetime.utcnow() - timedelta(days=window_days)
        window_df = df[df["event_date"] >= cutoff]

        if window_df.empty:
            return None

        type_counts = window_df["event_type"].value_counts().to_dict()

        # Geographic hotspot detection: cluster events and find high-density areas
        hotspots = []
        if not window_df.empty:
            location_counts = (
                window_df.groupby(["location", "latitude", "longitude"])
                .agg(event_count=("event_id", "count"), total_fatalities=("fatalities", "sum"))
                .reset_index()
                .sort_values("event_count", ascending=False)
                .head(10)
            )
            for _, loc_row in location_counts.iterrows():
                hotspots.append(
                    {
                        "location": loc_row["location"],
                        "latitude": loc_row["latitude"],
                        "longitude": loc_row["longitude"],
                        "event_count": int(loc_row["event_count"]),
                        "fatalities": int(loc_row["total_fatalities"]),
                    }
                )

        # Escalation score: normalized weighted severity over the window
        total_weighted_severity = window_df["weighted_severity"].sum()
        # Normalize by window length and historical baseline
        escalation_score = min(
            total_weighted_severity / (window_days * 2.0), 1.0
        )

        country_info = window_df.iloc[0]
        return ConflictSnapshot(
            country=country_info["country"],
            iso3=country_info.get("iso3", ""),
            window_start=cutoff,
            window_end=datetime.utcnow(),
            total_events=len(window_df),
            total_fatalities=int(window_df["fatalities"].sum()),
            battles=type_counts.get(EventType.BATTLES, 0),
            explosions=type_counts.get(EventType.EXPLOSIONS_REMOTE_VIOLENCE, 0),
            violence_against_civilians=type_counts.get(
                EventType.VIOLENCE_AGAINST_CIVILIANS, 0
            ),
            riots=type_counts.get(EventType.RIOTS, 0),
            protests=type_counts.get(EventType.PROTESTS, 0),
            strategic_developments=type_counts.get(
                EventType.STRATEGIC_DEVELOPMENTS, 0
            ),
            avg_severity=float(window_df["weighted_severity"].mean()),
            escalation_score=escalation_score,
            hotspot_locations=hotspots,
        )

    def compute_escalation_trend(
        self,
        country_key: str,
        short_window: int = 7,
        long_window: int = 30,
    ) -> dict[str, Any] | None:
        """Detect whether conflict is escalating or de-escalating.

        Compares short-window event rate/severity to long-window baseline.
        """
        df = self._event_cache.get(country_key)
        if df is None or df.empty:
            return None

        now = datetime.utcnow()
        short_cutoff = now - timedelta(days=short_window)
        long_cutoff = now - timedelta(days=long_window)

        short_df = df[df["event_date"] >= short_cutoff]
        long_df = df[df["event_date"] >= long_cutoff]

        if long_df.empty:
            return None

        # Daily event rates
        short_rate = len(short_df) / max(short_window, 1)
        long_rate = len(long_df) / max(long_window, 1)

        # Daily fatality rates
        short_fatality_rate = short_df["fatalities"].sum() / max(short_window, 1)
        long_fatality_rate = long_df["fatalities"].sum() / max(long_window, 1)

        # Severity trend
        short_severity = short_df["weighted_severity"].mean() if len(short_df) > 0 else 0
        long_severity = long_df["weighted_severity"].mean() if len(long_df) > 0 else 0

        # Escalation ratio: >1 means escalating
        event_ratio = short_rate / long_rate if long_rate > 0 else 0
        fatality_ratio = (
            short_fatality_rate / long_fatality_rate if long_fatality_rate > 0 else 0
        )
        severity_ratio = short_severity / long_severity if long_severity > 0 else 0

        composite_ratio = (event_ratio * 0.3 + fatality_ratio * 0.4 + severity_ratio * 0.3)

        if composite_ratio > 1.5:
            trend = "escalating_rapidly"
        elif composite_ratio > 1.1:
            trend = "escalating"
        elif composite_ratio > 0.9:
            trend = "stable"
        elif composite_ratio > 0.5:
            trend = "de_escalating"
        else:
            trend = "de_escalating_rapidly"

        return {
            "country_key": country_key,
            "trend": trend,
            "composite_ratio": composite_ratio,
            "event_rate_ratio": event_ratio,
            "fatality_rate_ratio": fatality_ratio,
            "severity_ratio": severity_ratio,
            "short_window_events": len(short_df),
            "long_window_events": len(long_df),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def detect_all_escalations(self) -> list[dict[str, Any]]:
        """Check all monitored countries for escalation trends."""
        results = []
        for entry in self.watched_countries:
            key = entry["iso3"] or entry["country"]
            trend = self.compute_escalation_trend(key)
            if trend:
                results.append(trend)
        return results
