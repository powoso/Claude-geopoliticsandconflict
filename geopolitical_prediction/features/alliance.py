"""Alliance network analysis for mutual defense treaty activation probability.

Models alliance networks as graphs to compute:
  - Probability that allies intervene in a conflict
  - Credibility of extended deterrence
  - Coalition formation likelihood
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Alliance:
    """A bilateral or multilateral defense alliance."""

    name: str
    members: list[str]
    alliance_type: str  # defense_pact, entente, neutrality, nonaggression
    signed_year: int
    active: bool = True
    article_5_equivalent: bool = False  # mutual defense obligation
    credibility_score: float = 0.5  # 0-1, how likely invocation leads to action


@dataclass
class AllianceNetworkAnalysis:
    """Analysis of a country's alliance network in context of a crisis."""

    country: str
    total_allies: int
    defense_pact_allies: int
    major_power_allies: list[str]
    collective_military_spending: float  # relative to adversary
    network_density: float  # how interconnected allies are
    weakest_link: str | None  # ally least likely to honor commitment
    estimated_intervention_probability: float


# Major alliances and their characteristics
ALLIANCES: list[Alliance] = [
    Alliance(
        name="NATO",
        members=[
            "United States", "United Kingdom", "France", "Germany", "Canada",
            "Italy", "Spain", "Turkey", "Poland", "Netherlands", "Belgium",
            "Norway", "Denmark", "Portugal", "Czech Republic", "Romania",
            "Bulgaria", "Hungary", "Slovakia", "Slovenia", "Croatia",
            "Albania", "Montenegro", "North Macedonia", "Lithuania",
            "Latvia", "Estonia", "Luxembourg", "Iceland", "Greece",
            "Finland", "Sweden",
        ],
        alliance_type="defense_pact",
        signed_year=1949,
        active=True,
        article_5_equivalent=True,
        credibility_score=0.85,
    ),
    Alliance(
        name="Five Eyes",
        members=["United States", "United Kingdom", "Canada", "Australia", "New Zealand"],
        alliance_type="intelligence_sharing",
        signed_year=1941,
        active=True,
        article_5_equivalent=False,
        credibility_score=0.90,
    ),
    Alliance(
        name="AUKUS",
        members=["United States", "United Kingdom", "Australia"],
        alliance_type="defense_pact",
        signed_year=2021,
        active=True,
        article_5_equivalent=False,
        credibility_score=0.80,
    ),
    Alliance(
        name="US-Japan Alliance",
        members=["United States", "Japan"],
        alliance_type="defense_pact",
        signed_year=1951,
        active=True,
        article_5_equivalent=True,
        credibility_score=0.80,
    ),
    Alliance(
        name="US-South Korea Alliance",
        members=["United States", "South Korea"],
        alliance_type="defense_pact",
        signed_year=1953,
        active=True,
        article_5_equivalent=True,
        credibility_score=0.85,
    ),
    Alliance(
        name="CSTO",
        members=["Russia", "Belarus", "Armenia", "Kazakhstan", "Kyrgyzstan", "Tajikistan"],
        alliance_type="defense_pact",
        signed_year=2002,
        active=True,
        article_5_equivalent=True,
        credibility_score=0.35,
    ),
    Alliance(
        name="SCO",
        members=["China", "Russia", "India", "Pakistan", "Kazakhstan",
                 "Kyrgyzstan", "Tajikistan", "Uzbekistan", "Iran"],
        alliance_type="entente",
        signed_year=2001,
        active=True,
        article_5_equivalent=False,
        credibility_score=0.20,
    ),
    Alliance(
        name="GCC",
        members=["Saudi Arabia", "UAE", "Kuwait", "Qatar", "Bahrain", "Oman"],
        alliance_type="defense_pact",
        signed_year=1981,
        active=True,
        article_5_equivalent=False,
        credibility_score=0.40,
    ),
]

# Approximate relative military spending (for weighting)
MILITARY_SPENDING_RELATIVE: dict[str, float] = {
    "United States": 100.0,
    "China": 30.0,
    "Russia": 10.0,
    "India": 8.0,
    "United Kingdom": 7.5,
    "Saudi Arabia": 7.0,
    "Germany": 6.5,
    "France": 6.0,
    "Japan": 5.5,
    "South Korea": 5.0,
    "Australia": 3.5,
    "Turkey": 3.0,
    "Israel": 2.5,
    "Italy": 3.5,
    "Canada": 2.5,
    "Poland": 2.5,
}


class AllianceNetworkAnalyzer:
    """Analyze alliance networks for intervention probability estimation."""

    def __init__(self, alliances: list[Alliance] | None = None):
        self.alliances = alliances or ALLIANCES
        self.graph = self._build_network()

    def _build_network(self) -> nx.Graph:
        """Build alliance network graph."""
        G = nx.Graph()

        for alliance in self.alliances:
            if not alliance.active:
                continue
            for i, member_a in enumerate(alliance.members):
                G.add_node(member_a)
                for member_b in alliance.members[i + 1:]:
                    if G.has_edge(member_a, member_b):
                        # Upgrade edge if stronger alliance type
                        existing = G[member_a][member_b]
                        if alliance.credibility_score > existing.get("credibility", 0):
                            G[member_a][member_b]["credibility"] = alliance.credibility_score
                            G[member_a][member_b]["alliance"] = alliance.name
                            G[member_a][member_b]["defense_pact"] = (
                                alliance.article_5_equivalent
                            )
                    else:
                        G.add_edge(
                            member_a,
                            member_b,
                            alliance=alliance.name,
                            credibility=alliance.credibility_score,
                            defense_pact=alliance.article_5_equivalent,
                        )

        return G

    def get_allies(self, country: str) -> list[dict[str, Any]]:
        """Get all allies of a country with alliance details."""
        if country not in self.graph:
            return []

        allies = []
        for neighbor in self.graph.neighbors(country):
            edge = self.graph[country][neighbor]
            allies.append(
                {
                    "ally": neighbor,
                    "alliance": edge.get("alliance", ""),
                    "credibility": edge.get("credibility", 0),
                    "defense_pact": edge.get("defense_pact", False),
                    "military_weight": MILITARY_SPENDING_RELATIVE.get(neighbor, 1.0),
                }
            )
        return sorted(allies, key=lambda x: x["military_weight"], reverse=True)

    def estimate_intervention_probability(
        self,
        defender: str,
        attacker: str,
    ) -> AllianceNetworkAnalysis:
        """Estimate probability that allies intervene if attacker targets defender.

        Considers:
        - Alliance type (defense pact vs entente)
        - Historical credibility
        - Military capability of allies
        - Whether attacker is also allied with potential interveners
        """
        allies = self.get_allies(defender)
        if not allies:
            return AllianceNetworkAnalysis(
                country=defender,
                total_allies=0,
                defense_pact_allies=0,
                major_power_allies=[],
                collective_military_spending=0,
                network_density=0,
                weakest_link=None,
                estimated_intervention_probability=0.05,
            )

        # Filter out allies that are also allied with the attacker
        # (they may be reluctant to pick a side)
        attacker_allies = {a["ally"] for a in self.get_allies(attacker)}

        defense_pact_allies = [a for a in allies if a["defense_pact"]]
        major_powers = [
            a["ally"]
            for a in allies
            if a["military_weight"] >= 5.0
        ]

        total_allied_military = sum(a["military_weight"] for a in allies)
        attacker_military = MILITARY_SPENDING_RELATIVE.get(attacker, 1.0)

        # Intervention probability factors
        # 1. Alliance credibility (average of defense pact credibilities)
        if defense_pact_allies:
            avg_credibility = sum(a["credibility"] for a in defense_pact_allies) / len(
                defense_pact_allies
            )
        else:
            avg_credibility = sum(a["credibility"] for a in allies) / len(allies) * 0.5

        # 2. Military capability ratio (allies vs attacker)
        capability_factor = min(total_allied_military / (attacker_military + 1), 1.0)

        # 3. Conflict of interest (allies also allied with attacker)
        conflicted_allies = [a for a in allies if a["ally"] in attacker_allies]
        conflict_penalty = len(conflicted_allies) / max(len(allies), 1) * 0.3

        # 4. Major power involvement boost
        major_power_boost = 0.1 * len(major_powers)

        intervention_prob = (
            avg_credibility * 0.5
            + capability_factor * 0.2
            + major_power_boost
            - conflict_penalty
        )
        intervention_prob = max(0.0, min(1.0, intervention_prob))

        # Network density (how interconnected the defender's allies are)
        ally_names = [a["ally"] for a in allies]
        subgraph = self.graph.subgraph([defender] + ally_names)
        density = nx.density(subgraph) if len(subgraph) > 1 else 0

        # Weakest link: ally with lowest credibility that holds a defense pact
        weakest = None
        if defense_pact_allies:
            weakest_ally = min(defense_pact_allies, key=lambda a: a["credibility"])
            weakest = weakest_ally["ally"]

        return AllianceNetworkAnalysis(
            country=defender,
            total_allies=len(allies),
            defense_pact_allies=len(defense_pact_allies),
            major_power_allies=major_powers,
            collective_military_spending=total_allied_military / attacker_military,
            network_density=density,
            weakest_link=weakest,
            estimated_intervention_probability=intervention_prob,
        )

    def find_flash_points(self) -> list[dict[str, Any]]:
        """Identify potential conflict flash points in the alliance network.

        Looks for country pairs that are:
        1. NOT allied with each other
        2. Have strong, competing alliance networks
        """
        countries = list(self.graph.nodes())
        flash_points = []

        for i, country_a in enumerate(countries):
            for country_b in countries[i + 1:]:
                if self.graph.has_edge(country_a, country_b):
                    continue  # allies, not flash point

                allies_a = set(self.graph.neighbors(country_a))
                allies_b = set(self.graph.neighbors(country_b))

                # Countries with overlapping allies face complex dynamics
                shared_allies = allies_a & allies_b
                mil_a = MILITARY_SPENDING_RELATIVE.get(country_a, 0)
                mil_b = MILITARY_SPENDING_RELATIVE.get(country_b, 0)

                if mil_a >= 5.0 and mil_b >= 5.0:
                    flash_points.append(
                        {
                            "country_a": country_a,
                            "country_b": country_b,
                            "military_a": mil_a,
                            "military_b": mil_b,
                            "shared_allies": list(shared_allies),
                            "tension_score": (mil_a + mil_b) / 100,
                        }
                    )

        return sorted(flash_points, key=lambda x: x["tension_score"], reverse=True)
