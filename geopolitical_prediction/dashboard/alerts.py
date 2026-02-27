"""Real-time escalation alerting system.

Monitors all data pipelines and triggers alerts when:
  - Conflict escalation scores cross thresholds
  - Sentiment shifts exceed z-score thresholds
  - Commodity stress signals fire
  - Social media volume spikes on crisis keywords
  - Model-market probability divergences appear
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    ESCALATION = "escalation"
    SENTIMENT_SHIFT = "sentiment_shift"
    COMMODITY_STRESS = "commodity_stress"
    SOCIAL_MEDIA_SPIKE = "social_media_spike"
    MODEL_MARKET_DIVERGENCE = "model_market_divergence"
    SATELLITE_CHANGE = "satellite_change"
    SHIPPING_ANOMALY = "shipping_anomaly"


@dataclass
class Alert:
    """A geopolitical escalation alert."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    country: str
    title: str
    message: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    expires_at: datetime | None = None


@dataclass
class AlertRule:
    """A rule that defines when an alert should fire."""

    name: str
    alert_type: AlertType
    condition: Callable[..., bool]
    severity_fn: Callable[..., AlertSeverity]
    message_fn: Callable[..., str]
    cooldown_minutes: int = 60  # don't re-fire within this window


class AlertEngine:
    """Engine that evaluates conditions and fires alerts."""

    def __init__(self, escalation_threshold: float = 0.7) -> None:
        self.escalation_threshold = escalation_threshold
        self.rules: list[AlertRule] = self._default_rules()
        self.active_alerts: list[Alert] = []
        self._last_fired: dict[str, datetime] = {}
        self._subscribers: list[Callable[[Alert], None]] = []
        self._alert_counter = 0

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback to receive alerts."""
        self._subscribers.append(callback)

    def _notify(self, alert: Alert) -> None:
        for callback in self._subscribers:
            try:
                callback(alert)
            except Exception:
                logger.exception("Alert subscriber callback failed")

    def _default_rules(self) -> list[AlertRule]:
        return [
            AlertRule(
                name="conflict_escalation",
                alert_type=AlertType.ESCALATION,
                condition=lambda data: data.get("escalation_score", 0) >= self.escalation_threshold,
                severity_fn=lambda data: (
                    AlertSeverity.CRITICAL
                    if data.get("escalation_score", 0) >= 0.9
                    else AlertSeverity.WARNING
                ),
                message_fn=lambda data: (
                    f"Conflict escalation score for {data.get('country', '?')} "
                    f"reached {data.get('escalation_score', 0):.2f} "
                    f"(threshold: {self.escalation_threshold:.2f}). "
                    f"Trend: {data.get('trend', 'unknown')}"
                ),
            ),
            AlertRule(
                name="sentiment_deterioration",
                alert_type=AlertType.SENTIMENT_SHIFT,
                condition=lambda data: data.get("z_score", 0) <= -2.0,
                severity_fn=lambda data: (
                    AlertSeverity.CRITICAL
                    if data.get("z_score", 0) <= -3.0
                    else AlertSeverity.WARNING
                ),
                message_fn=lambda data: (
                    f"Sentiment deterioration for {data.get('key', '?')}: "
                    f"z-score={data.get('z_score', 0):.1f}, "
                    f"direction={data.get('direction', 'unknown')}"
                ),
            ),
            AlertRule(
                name="commodity_stress",
                alert_type=AlertType.COMMODITY_STRESS,
                condition=lambda data: data.get("is_stressed", False),
                severity_fn=lambda data: (
                    AlertSeverity.WARNING
                    if abs(data.get("z_score", 0)) < 3.0
                    else AlertSeverity.CRITICAL
                ),
                message_fn=lambda data: (
                    f"Commodity stress: {data.get('commodity', '?')} "
                    f"{data.get('direction', '?')} (z={data.get('z_score', 0):.1f})"
                ),
            ),
            AlertRule(
                name="social_media_spike",
                alert_type=AlertType.SOCIAL_MEDIA_SPIKE,
                condition=lambda data: data.get("is_spike", False),
                severity_fn=lambda _data: AlertSeverity.INFO,
                message_fn=lambda data: (
                    f"Social media volume spike for '{data.get('keyword', '?')}': "
                    f"{data.get('tweet_count', 0)} tweets, z={data.get('volume_z_score', 0):.1f}"
                ),
            ),
            AlertRule(
                name="model_market_divergence",
                alert_type=AlertType.MODEL_MARKET_DIVERGENCE,
                condition=lambda data: data.get("significant_divergence", False),
                severity_fn=lambda _data: AlertSeverity.INFO,
                message_fn=lambda data: (
                    f"Model-market divergence for '{data.get('query', '?')}': "
                    f"model={data.get('model_probability', 0):.1%}, "
                    f"market={data.get('market_probability', 0):.1%}"
                ),
            ),
        ]

    def evaluate(
        self,
        alert_type: AlertType,
        country: str,
        data: dict[str, Any],
    ) -> Alert | None:
        """Evaluate data against rules and fire alert if conditions are met."""
        for rule in self.rules:
            if rule.alert_type != alert_type:
                continue

            # Check cooldown
            last = self._last_fired.get(f"{rule.name}_{country}")
            if last:
                elapsed = (datetime.utcnow() - last).total_seconds() / 60
                if elapsed < rule.cooldown_minutes:
                    continue

            try:
                if rule.condition(data):
                    self._alert_counter += 1
                    alert = Alert(
                        alert_id=f"alert_{self._alert_counter:06d}",
                        alert_type=alert_type,
                        severity=rule.severity_fn(data),
                        country=country,
                        title=rule.name.replace("_", " ").title(),
                        message=rule.message_fn(data),
                        data=data,
                    )
                    self.active_alerts.append(alert)
                    self._last_fired[f"{rule.name}_{country}"] = datetime.utcnow()
                    self._notify(alert)
                    logger.warning(
                        "ALERT [%s] %s: %s", alert.severity.value, alert.title, alert.message
                    )
                    return alert
            except Exception:
                logger.exception("Alert rule evaluation failed: %s", rule.name)

        return None

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        country: str | None = None,
    ) -> list[Alert]:
        """Get active (unacknowledged) alerts with optional filters."""
        alerts = [a for a in self.active_alerts if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if country:
            alerts = [a for a in alerts if a.country == country]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
