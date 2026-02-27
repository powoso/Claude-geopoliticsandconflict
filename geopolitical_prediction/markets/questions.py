"""Predefined geopolitical prediction market questions.

Defines the structure of questions the system can forecast:
  - "Will X country invade Y?"
  - "Will sanctions be imposed on X?"
  - "Will trade deal be signed by date?"
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from geopolitical_prediction.models.ensemble import MarketQuestion

# Active prediction market questions
ACTIVE_QUESTIONS: list[MarketQuestion] = [
    # Invasion / military conflict questions
    MarketQuestion(
        question_id="china_taiwan_invasion_2026",
        question_text="Will China launch a military invasion of Taiwan by end of 2026?",
        question_type="invasion",
        country_a="China",
        country_b="Taiwan",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="military_buildup_to_invasion",
    ),
    MarketQuestion(
        question_id="russia_nato_direct_conflict_2026",
        question_text="Will there be direct military conflict between Russia and a NATO member by end of 2026?",
        question_type="invasion",
        country_a="Russia",
        country_b="NATO",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="interstate_crisis_to_war",
    ),
    MarketQuestion(
        question_id="north_korea_south_korea_conflict_2026",
        question_text="Will North Korea initiate military action against South Korea by end of 2026?",
        question_type="invasion",
        country_a="North Korea",
        country_b="South Korea",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="military_buildup_to_invasion",
    ),
    MarketQuestion(
        question_id="india_pakistan_conflict_2026",
        question_text="Will there be a significant military confrontation between India and Pakistan by end of 2026?",
        question_type="invasion",
        country_a="India",
        country_b="Pakistan",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="interstate_crisis_to_war",
    ),

    # Sanctions questions
    MarketQuestion(
        question_id="us_china_tech_sanctions_expansion_2026",
        question_text="Will the US significantly expand technology sanctions on China by mid-2026?",
        question_type="sanctions",
        country_a="United States",
        country_b="China",
        resolution_date=datetime(2026, 6, 30),
        base_rate_key="sanctions_threat_to_imposition",
    ),
    MarketQuestion(
        question_id="eu_russia_sanctions_12th_package",
        question_text="Will the EU adopt a new major sanctions package against Russia by end of 2026?",
        question_type="sanctions",
        country_a="EU",
        country_b="Russia",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="sanctions_threat_to_imposition",
    ),
    MarketQuestion(
        question_id="iran_sanctions_snap_back_2026",
        question_text="Will nuclear-related sanctions on Iran be formally reimposed by end of 2026?",
        question_type="sanctions",
        country_a="United States",
        country_b="Iran",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="sanctions_threat_to_imposition",
    ),

    # Trade deal questions
    MarketQuestion(
        question_id="us_uk_trade_deal_2026",
        question_text="Will the US and UK sign a comprehensive trade deal by end of 2026?",
        question_type="trade_deal",
        country_a="United States",
        country_b="United Kingdom",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="trade_deal_completion",
    ),
    MarketQuestion(
        question_id="rcep_india_accession_2027",
        question_text="Will India join RCEP by end of 2027?",
        question_type="trade_deal",
        country_a="India",
        country_b=None,
        resolution_date=datetime(2027, 12, 31),
        base_rate_key="trade_deal_completion",
    ),

    # Ceasefire questions
    MarketQuestion(
        question_id="ukraine_ceasefire_2026",
        question_text="Will a formal ceasefire between Russia and Ukraine be established by end of 2026?",
        question_type="ceasefire",
        country_a="Russia",
        country_b="Ukraine",
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="ceasefire_holds_1yr",
    ),
    MarketQuestion(
        question_id="sudan_ceasefire_2026",
        question_text="Will a lasting ceasefire be achieved in Sudan's civil war by end of 2026?",
        question_type="ceasefire",
        country_a="Sudan",
        country_b=None,
        resolution_date=datetime(2026, 12, 31),
        base_rate_key="ceasefire_holds_1yr",
    ),
]


def get_question(question_id: str) -> MarketQuestion | None:
    """Get a question by ID."""
    for q in ACTIVE_QUESTIONS:
        if q.question_id == question_id:
            return q
    return None


def get_questions_by_type(question_type: str) -> list[MarketQuestion]:
    """Get all questions of a given type."""
    return [q for q in ACTIVE_QUESTIONS if q.question_type == question_type]


def get_questions_by_country(country: str) -> list[MarketQuestion]:
    """Get all questions involving a specific country."""
    return [
        q for q in ACTIVE_QUESTIONS
        if q.country_a == country or q.country_b == country
    ]


def create_custom_question(
    question_id: str,
    question_text: str,
    question_type: str,
    country_a: str,
    country_b: str | None = None,
    resolution_date: datetime | None = None,
    base_rate_key: str = "interstate_crisis_to_war",
) -> MarketQuestion:
    """Create a custom prediction market question."""
    return MarketQuestion(
        question_id=question_id,
        question_text=question_text,
        question_type=question_type,
        country_a=country_a,
        country_b=country_b,
        resolution_date=resolution_date,
        base_rate_key=base_rate_key,
    )
