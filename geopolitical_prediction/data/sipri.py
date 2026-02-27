"""SIPRI Arms Transfer Database client.

Tracks international transfers of major conventional weapons.
Arms flows indicate alliance strength, military buildup, and
potential aggression preparation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SIPRI Arms Transfers Database (public, free access)
SIPRI_TRADE_REGISTER_URL = "https://armstrade.sipri.org/armstrade/page/trade_register.php"


@dataclass
class ArmsTransfer:
    """A single arms transfer record."""

    supplier: str
    recipient: str
    year_ordered: int
    year_delivered: int | None
    weapon_description: str
    weapon_designation: str
    quantity: int
    status: str  # ordered, delivered, in production
    tiv_value: float  # Trend Indicator Value (SIPRI's unit of measurement)
    comments: str = ""


class SIPRIClient:
    """Client for SIPRI arms transfer data.

    Since SIPRI doesn't offer a formal REST API, this module provides
    utilities for working with exported CSV data and structured analysis.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self._transfer_cache: pd.DataFrame | None = None

    def load_transfers(self, csv_path: str) -> pd.DataFrame:
        """Load SIPRI arms transfer data from exported CSV.

        Download from: https://armstrade.sipri.org/armstrade/page/trade_register.php
        Export as CSV after selecting desired filters.
        """
        try:
            df = pd.read_csv(csv_path)
            self._transfer_cache = df
            logger.info("Loaded %d SIPRI transfer records from %s", len(df), csv_path)
            return df
        except Exception:
            logger.exception("Failed to load SIPRI data from %s", csv_path)
            return pd.DataFrame()

    def get_supplier_profile(
        self,
        supplier: str,
        df: pd.DataFrame | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Analyze arms export patterns for a supplier country.

        Returns top recipients, weapon categories, and total TIV volume.
        """
        data = df if df is not None else self._transfer_cache
        if data is None or data.empty:
            return {}

        filtered = data[data["supplier"].str.contains(supplier, case=False, na=False)]
        if year_range:
            filtered = filtered[
                filtered["year_delivered"].between(year_range[0], year_range[1])
            ]

        if filtered.empty:
            return {"supplier": supplier, "total_transfers": 0}

        top_recipients = (
            filtered.groupby("recipient")["tiv_value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        return {
            "supplier": supplier,
            "total_transfers": len(filtered),
            "total_tiv": float(filtered["tiv_value"].sum()),
            "top_recipients": top_recipients,
            "year_range": year_range,
        }

    def get_recipient_profile(
        self,
        recipient: str,
        df: pd.DataFrame | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Analyze arms import patterns for a recipient country.

        Sudden increases in arms imports can signal military buildup.
        """
        data = df if df is not None else self._transfer_cache
        if data is None or data.empty:
            return {}

        filtered = data[data["recipient"].str.contains(recipient, case=False, na=False)]
        if year_range:
            filtered = filtered[
                filtered["year_delivered"].between(year_range[0], year_range[1])
            ]

        if filtered.empty:
            return {"recipient": recipient, "total_transfers": 0}

        top_suppliers = (
            filtered.groupby("supplier")["tiv_value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        weapon_types = filtered["weapon_description"].value_counts().head(10).to_dict()

        return {
            "recipient": recipient,
            "total_transfers": len(filtered),
            "total_tiv": float(filtered["tiv_value"].sum()),
            "top_suppliers": top_suppliers,
            "weapon_types": weapon_types,
        }

    def detect_buildup(
        self,
        country: str,
        df: pd.DataFrame | None = None,
        baseline_years: tuple[int, int] = (2010, 2019),
        recent_years: tuple[int, int] = (2020, 2025),
    ) -> dict[str, Any]:
        """Detect significant changes in arms import volume.

        Compares recent TIV volume to historical baseline.
        A spike may indicate military buildup preparation.
        """
        data = df if df is not None else self._transfer_cache
        if data is None or data.empty:
            return {"country": country, "buildup_detected": False}

        recipient_data = data[
            data["recipient"].str.contains(country, case=False, na=False)
        ]

        baseline = recipient_data[
            recipient_data["year_delivered"].between(baseline_years[0], baseline_years[1])
        ]
        recent = recipient_data[
            recipient_data["year_delivered"].between(recent_years[0], recent_years[1])
        ]

        baseline_years_count = baseline_years[1] - baseline_years[0] + 1
        recent_years_count = recent_years[1] - recent_years[0] + 1

        baseline_annual_tiv = (
            baseline["tiv_value"].sum() / baseline_years_count if baseline_years_count else 0
        )
        recent_annual_tiv = (
            recent["tiv_value"].sum() / recent_years_count if recent_years_count else 0
        )

        ratio = recent_annual_tiv / baseline_annual_tiv if baseline_annual_tiv > 0 else 0

        return {
            "country": country,
            "baseline_annual_tiv": float(baseline_annual_tiv),
            "recent_annual_tiv": float(recent_annual_tiv),
            "change_ratio": float(ratio),
            "buildup_detected": ratio > 1.5,
            "buildup_severity": "high" if ratio > 2.5 else "moderate" if ratio > 1.5 else "low",
        }

    def bilateral_arms_flow(
        self,
        country_a: str,
        country_b: str,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Analyze arms flows between two specific countries.

        Bidirectional arms trade suggests alliance; one-sided supply
        to a conflict party is an escalation indicator.
        """
        data = df if df is not None else self._transfer_cache
        if data is None or data.empty:
            return {}

        a_to_b = data[
            data["supplier"].str.contains(country_a, case=False, na=False)
            & data["recipient"].str.contains(country_b, case=False, na=False)
        ]
        b_to_a = data[
            data["supplier"].str.contains(country_b, case=False, na=False)
            & data["recipient"].str.contains(country_a, case=False, na=False)
        ]

        return {
            "pair": f"{country_a} <-> {country_b}",
            "a_to_b_tiv": float(a_to_b["tiv_value"].sum()) if not a_to_b.empty else 0,
            "b_to_a_tiv": float(b_to_a["tiv_value"].sum()) if not b_to_a.empty else 0,
            "a_to_b_transfers": len(a_to_b),
            "b_to_a_transfers": len(b_to_a),
            "bidirectional": len(a_to_b) > 0 and len(b_to_a) > 0,
        }
