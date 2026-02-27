"""UN General Assembly and Security Council voting records.

Tracks voting alignment, vetoes, and resolution patterns as leading
indicators of international coalition formation and diplomatic breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# UN Digital Library / voting data endpoints
UNGA_VOTES_URL = "https://dataverse.harvard.edu/api/access/datafile/:persistentId"
UNSC_VETOES_URL = "https://www.un.org/depts/dhl/resguide/scact_veto_table_en.htm"

# Voting alignment interpretation
VOTE_CODES: dict[int, str] = {
    1: "Yes",
    2: "Abstain",
    3: "No",
    8: "Absent",
    9: "Not a member",
}


@dataclass
class UNVoteRecord:
    """A single UN General Assembly vote record."""

    resolution_id: str
    session: int
    date: datetime
    short_description: str
    country: str
    country_code: str
    vote: int  # 1=Yes, 2=Abstain, 3=No, 8=Absent, 9=Not member
    important_vote: bool = False
    topic_codes: list[str] = field(default_factory=list)


@dataclass
class VotingAlignment:
    """Bilateral voting alignment score between two countries."""

    country_a: str
    country_b: str
    alignment_score: float  # 0 to 1, where 1 = identical voting
    total_votes: int
    agree_count: int
    disagree_count: int
    period_start: datetime | None = None
    period_end: datetime | None = None


@dataclass
class SecurityCouncilVeto:
    """A Security Council veto record."""

    date: datetime
    draft_resolution: str
    subject: str
    vetoing_countries: list[str]


class UNVotingClient:
    """Client for fetching and analyzing UN voting data.

    Uses the Erik Voeten UN General Assembly Voting Dataset
    (hosted on Harvard Dataverse) and supplementary SC data.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self._vote_cache: pd.DataFrame | None = None

    def load_unga_votes(self, csv_path: str | None = None) -> pd.DataFrame:
        """Load UN General Assembly voting data.

        Can load from a local CSV or attempt to fetch from Dataverse.
        The dataset contains country-level votes on all UNGA resolutions.
        """
        if self._vote_cache is not None:
            return self._vote_cache

        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                self._vote_cache = df
                return df
            except Exception:
                logger.exception("Failed to load UNGA votes from %s", csv_path)
                return pd.DataFrame()

        # Return empty if no local file provided (dataset is large)
        logger.warning(
            "No UNGA vote CSV provided. Download from Harvard Dataverse: "
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/12379"
        )
        return pd.DataFrame()

    def compute_bilateral_alignment(
        self,
        votes_df: pd.DataFrame,
        country_a_code: str,
        country_b_code: str,
        session_range: tuple[int, int] | None = None,
    ) -> VotingAlignment:
        """Compute voting alignment between two countries.

        Uses the agreement score: proportion of votes where both countries
        voted the same way (excluding absences and non-membership).
        """
        df = votes_df.copy()

        if session_range:
            df = df[df["session"].between(session_range[0], session_range[1])]

        # Filter to relevant countries and valid votes
        a_votes = df[df["ccode"] == country_a_code][["rcid", "vote"]].rename(
            columns={"vote": "vote_a"}
        )
        b_votes = df[df["ccode"] == country_b_code][["rcid", "vote"]].rename(
            columns={"vote": "vote_b"}
        )

        merged = a_votes.merge(b_votes, on="rcid")
        # Exclude absences and non-membership
        valid = merged[merged["vote_a"].isin([1, 2, 3]) & merged["vote_b"].isin([1, 2, 3])]

        if valid.empty:
            return VotingAlignment(
                country_a=country_a_code,
                country_b=country_b_code,
                alignment_score=0.0,
                total_votes=0,
                agree_count=0,
                disagree_count=0,
            )

        agree = (valid["vote_a"] == valid["vote_b"]).sum()
        total = len(valid)
        disagree = total - agree

        return VotingAlignment(
            country_a=country_a_code,
            country_b=country_b_code,
            alignment_score=agree / total if total > 0 else 0.0,
            total_votes=total,
            agree_count=int(agree),
            disagree_count=int(disagree),
        )

    def compute_alignment_trend(
        self,
        votes_df: pd.DataFrame,
        country_a_code: str,
        country_b_code: str,
        window_sessions: int = 5,
    ) -> pd.DataFrame:
        """Compute voting alignment trend over time by session.

        Returns a DataFrame with alignment scores per session,
        useful for detecting diplomatic convergence/divergence.
        """
        if votes_df.empty:
            return pd.DataFrame()

        sessions = sorted(votes_df["session"].unique())
        rows = []
        for i in range(len(sessions)):
            start_idx = max(0, i - window_sessions + 1)
            session_range = (sessions[start_idx], sessions[i])
            alignment = self.compute_bilateral_alignment(
                votes_df, country_a_code, country_b_code, session_range
            )
            rows.append(
                {
                    "session": sessions[i],
                    "alignment_score": alignment.alignment_score,
                    "total_votes": alignment.total_votes,
                    "window_start": session_range[0],
                    "window_end": session_range[1],
                }
            )
        return pd.DataFrame(rows)

    def find_coalition_blocs(
        self,
        votes_df: pd.DataFrame,
        session: int | None = None,
        threshold: float = 0.8,
    ) -> list[list[str]]:
        """Identify voting blocs (groups of countries that vote similarly).

        Uses simple hierarchical clustering on voting patterns.
        Countries with alignment > threshold are grouped together.
        """
        import numpy as np

        if votes_df.empty:
            return []

        df = votes_df.copy()
        if session:
            df = df[df["session"] == session]

        # Pivot to country x resolution voting matrix
        pivot = df[df["vote"].isin([1, 2, 3])].pivot_table(
            index="ccode", columns="rcid", values="vote", fill_value=0
        )

        if pivot.empty or len(pivot) < 2:
            return []

        countries = list(pivot.index)
        n = len(countries)

        # Compute pairwise alignment
        alignment_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                common = (pivot.iloc[i] != 0) & (pivot.iloc[j] != 0)
                if common.sum() > 0:
                    agree = (
                        (pivot.iloc[i][common] == pivot.iloc[j][common]).sum()
                        / common.sum()
                    )
                    alignment_matrix[i, j] = agree
                    alignment_matrix[j, i] = agree

        # Simple greedy clustering
        visited = set()
        blocs = []
        for i in range(n):
            if i in visited:
                continue
            bloc = [countries[i]]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and alignment_matrix[i, j] >= threshold:
                    bloc.append(countries[j])
                    visited.add(j)
            if len(bloc) > 1:
                blocs.append(bloc)

        return blocs
