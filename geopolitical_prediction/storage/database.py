"""Database storage for events, predictions, and calibration data.

Uses SQLAlchemy for database-agnostic storage (SQLite for local,
PostgreSQL for production).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from geopolitical_prediction.config import DatabaseConfig, get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConflictEventRecord(Base):
    """Stored ACLED conflict event."""

    __tablename__ = "conflict_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, index=True)
    event_date = Column(DateTime, index=True)
    event_type = Column(String(100))
    sub_event_type = Column(String(100))
    actor1 = Column(String(500))
    actor2 = Column(String(500))
    country = Column(String(100), index=True)
    iso3 = Column(String(10), index=True)
    location = Column(String(500))
    latitude = Column(Float)
    longitude = Column(Float)
    fatalities = Column(Integer)
    severity_weight = Column(Float)
    notes = Column(Text)
    ingested_at = Column(DateTime, default=datetime.utcnow)


class SentimentRecord(Base):
    """Stored GDELT sentiment data point."""

    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    country_a = Column(String(100), index=True)
    country_b = Column(String(100), index=True)
    date = Column(DateTime, index=True)
    avg_tone = Column(Float)
    article_count = Column(Integer)
    tone_7d_avg = Column(Float, nullable=True)
    tone_30d_avg = Column(Float, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)


class PredictionRecord(Base):
    """Stored model prediction for a market question."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(100), index=True)
    question_text = Column(Text)
    question_type = Column(String(50))
    country_a = Column(String(100))
    country_b = Column(String(100), nullable=True)
    timestamp = Column(DateTime, index=True)
    probability = Column(Float)
    confidence = Column(Float)
    base_rate = Column(Float)
    escalation_prob = Column(Float)
    analog_prob = Column(Float)
    sentiment_signal = Column(Float)
    conflict_signal = Column(Float)
    market_price = Column(Float, nullable=True)
    model_market_divergence = Column(Float, nullable=True)
    signals_used = Column(JSON)
    reasoning = Column(JSON)


class AlertRecord(Base):
    """Stored escalation alert."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(100))  # escalation, sentiment_shift, market_divergence
    country = Column(String(100), index=True)
    severity = Column(String(20))  # info, warning, critical
    message = Column(Text)
    data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged = Column(Boolean, default=False)


class CalibrationRecord(Base):
    """Stored calibration data for model accuracy tracking."""

    __tablename__ = "calibration"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(100), index=True)
    predicted_probability = Column(Float)
    actual_outcome = Column(Boolean)
    resolution_date = Column(DateTime)
    brier_contribution = Column(Float)


class DatabaseManager:
    """Database connection and query manager."""

    def __init__(self, config: DatabaseConfig | None = None):
        cfg = config or get_settings().database
        self.engine = create_engine(cfg.url, echo=False)
        self.SessionFactory = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def get_session(self) -> Session:
        return self.SessionFactory()

    # ----- Conflict Events -----

    def store_conflict_events(self, events: list[dict[str, Any]]) -> int:
        """Store ACLED conflict events, skipping duplicates."""
        session = self.get_session()
        stored = 0
        try:
            for event in events:
                existing = (
                    session.query(ConflictEventRecord)
                    .filter_by(event_id=event.get("event_id"))
                    .first()
                )
                if not existing:
                    record = ConflictEventRecord(**event)
                    session.add(record)
                    stored += 1
            session.commit()
            logger.info("Stored %d new conflict events", stored)
        except Exception:
            session.rollback()
            logger.exception("Failed to store conflict events")
        finally:
            session.close()
        return stored

    def get_conflict_events(
        self,
        country: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Query stored conflict events."""
        session = self.get_session()
        try:
            query = session.query(ConflictEventRecord)
            if country:
                query = query.filter(ConflictEventRecord.country == country)
            if start_date:
                query = query.filter(ConflictEventRecord.event_date >= start_date)
            if end_date:
                query = query.filter(ConflictEventRecord.event_date <= end_date)
            query = query.order_by(ConflictEventRecord.event_date.desc()).limit(limit)

            results = []
            for record in query.all():
                results.append(
                    {
                        "event_id": record.event_id,
                        "event_date": record.event_date,
                        "event_type": record.event_type,
                        "country": record.country,
                        "location": record.location,
                        "latitude": record.latitude,
                        "longitude": record.longitude,
                        "fatalities": record.fatalities,
                    }
                )
            return results
        finally:
            session.close()

    # ----- Predictions -----

    def store_prediction(self, prediction: dict[str, Any]) -> None:
        """Store a model prediction."""
        session = self.get_session()
        try:
            record = PredictionRecord(**prediction)
            session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to store prediction")
        finally:
            session.close()

    def get_prediction_history(
        self,
        question_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get prediction history for a question."""
        session = self.get_session()
        try:
            records = (
                session.query(PredictionRecord)
                .filter_by(question_id=question_id)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": r.timestamp,
                    "probability": r.probability,
                    "confidence": r.confidence,
                    "market_price": r.market_price,
                }
                for r in records
            ]
        finally:
            session.close()

    # ----- Alerts -----

    def store_alert(self, alert: dict[str, Any]) -> None:
        """Store an escalation alert."""
        session = self.get_session()
        try:
            record = AlertRecord(**alert)
            session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to store alert")
        finally:
            session.close()

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get most recent alerts."""
        session = self.get_session()
        try:
            records = (
                session.query(AlertRecord)
                .order_by(AlertRecord.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "alert_type": r.alert_type,
                    "country": r.country,
                    "severity": r.severity,
                    "message": r.message,
                    "timestamp": r.timestamp,
                    "acknowledged": r.acknowledged,
                }
                for r in records
            ]
        finally:
            session.close()
