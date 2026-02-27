"""Domestic political incentive modeling for diversionary conflict prediction.

Models the "diversionary war" hypothesis: leaders facing domestic political
pressure may escalate foreign crises to rally public support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DomesticPoliticalState:
    """Domestic political conditions of a country's leadership."""

    country: str
    leader_name: str
    regime_type: str  # democracy, autocracy, hybrid
    approval_rating: float | None  # 0-100 if available
    months_to_election: int | None  # None if no scheduled election
    in_economic_recession: bool
    unemployment_rate: float | None
    inflation_rate: float | None
    ongoing_protests: bool
    protest_severity: float  # 0-1
    recent_scandal: bool
    years_in_power: int
    term_limited: bool | None  # True if cannot run again


@dataclass
class DiversionaryRiskScore:
    """Risk assessment for diversionary conflict behavior."""

    country: str
    timestamp: datetime
    # Component scores
    domestic_pressure_score: float  # 0-1, how much pressure leader faces
    electoral_incentive_score: float  # 0-1, incentive from upcoming election
    economic_distress_score: float  # 0-1, economic problems driving diversion
    # Composite
    diversionary_risk: float  # 0-1, overall likelihood of diversionary behavior
    risk_level: str  # low, moderate, high, very_high
    risk_factors: list[str]


def compute_domestic_pressure(state: DomesticPoliticalState) -> float:
    """Compute domestic political pressure score.

    Higher score = more pressure on leader, higher diversionary incentive.
    """
    pressure = 0.0
    factors_count = 0

    # Low approval is a strong pressure signal
    if state.approval_rating is not None:
        # Below 40% = high pressure, below 30% = very high
        approval_pressure = max(0, (50 - state.approval_rating) / 50)
        pressure += approval_pressure * 0.3
        factors_count += 1

    # Ongoing protests indicate public discontent
    if state.ongoing_protests:
        pressure += state.protest_severity * 0.2
        factors_count += 1

    # Recent scandal creates political vulnerability
    if state.recent_scandal:
        pressure += 0.15
        factors_count += 1

    # Economic distress
    if state.in_economic_recession:
        pressure += 0.15

    if state.inflation_rate is not None and state.inflation_rate > 5:
        pressure += min(state.inflation_rate / 50, 0.2)

    return min(pressure, 1.0)


def compute_electoral_incentive(state: DomesticPoliticalState) -> float:
    """Compute electoral incentive for diversionary action.

    Leaders facing elections with low approval have the strongest
    incentive for rally-around-the-flag effects.
    """
    if state.regime_type == "autocracy":
        return 0.1  # autocrats face less electoral pressure

    if state.months_to_election is None:
        return 0.0

    # Peak incentive: 3-12 months before election
    # Too close (< 3 months) may be too late; too far (> 18 months) less urgent
    if state.months_to_election < 3:
        proximity_factor = 0.5
    elif state.months_to_election <= 12:
        proximity_factor = 1.0 - (state.months_to_election - 3) / 18
    elif state.months_to_election <= 18:
        proximity_factor = 0.3
    else:
        proximity_factor = 0.1

    # Term-limited leaders have less incentive (no reelection to win)
    if state.term_limited:
        proximity_factor *= 0.3

    # Low approval amplifies electoral incentive
    approval_factor = 1.0
    if state.approval_rating is not None:
        if state.approval_rating < 40:
            approval_factor = 1.5
        elif state.approval_rating < 50:
            approval_factor = 1.2

    return min(proximity_factor * approval_factor, 1.0)


def compute_diversionary_risk(
    state: DomesticPoliticalState,
) -> DiversionaryRiskScore:
    """Compute overall diversionary conflict risk for a country.

    Combines domestic pressure, electoral incentives, and economic
    distress into a composite diversionary war risk score.
    """
    domestic_pressure = compute_domestic_pressure(state)
    electoral_incentive = compute_electoral_incentive(state)

    # Economic distress score
    economic_distress = 0.0
    if state.in_economic_recession:
        economic_distress += 0.4
    if state.unemployment_rate is not None and state.unemployment_rate > 8:
        economic_distress += min(state.unemployment_rate / 30, 0.3)
    if state.inflation_rate is not None and state.inflation_rate > 10:
        economic_distress += min(state.inflation_rate / 50, 0.3)
    economic_distress = min(economic_distress, 1.0)

    # Composite risk
    composite = (
        0.35 * domestic_pressure
        + 0.35 * electoral_incentive
        + 0.30 * economic_distress
    )

    # Regime type modifier
    if state.regime_type == "autocracy":
        # Autocrats have more freedom to act on diversionary impulses
        composite *= 1.2
    elif state.regime_type == "democracy":
        # Democratic checks constrain diversionary war somewhat
        composite *= 0.9

    composite = min(composite, 1.0)

    # Risk level classification
    if composite > 0.7:
        risk_level = "very_high"
    elif composite > 0.5:
        risk_level = "high"
    elif composite > 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Enumerate specific risk factors
    risk_factors = []
    if state.approval_rating is not None and state.approval_rating < 35:
        risk_factors.append(f"Very low approval rating: {state.approval_rating}%")
    if state.months_to_election is not None and state.months_to_election < 12:
        risk_factors.append(f"Election in {state.months_to_election} months")
    if state.in_economic_recession:
        risk_factors.append("Economy in recession")
    if state.ongoing_protests:
        risk_factors.append("Ongoing domestic protests")
    if state.recent_scandal:
        risk_factors.append("Recent political scandal")

    return DiversionaryRiskScore(
        country=state.country,
        timestamp=datetime.utcnow(),
        domestic_pressure_score=domestic_pressure,
        electoral_incentive_score=electoral_incentive,
        economic_distress_score=economic_distress,
        diversionary_risk=composite,
        risk_level=risk_level,
        risk_factors=risk_factors,
    )
