"""Ship tracking data (AIS) for trade and sanctions monitoring.

Vessel traffic patterns reveal:
  - Sanctions evasion (ship-to-ship transfers, AIS dark activity)
  - Trade route disruptions
  - Naval force deployments
  - Blockade indicators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from geopolitical_prediction.config import MarineTrafficConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class VesselPosition:
    """A single AIS vessel position report."""

    mmsi: str  # Maritime Mobile Service Identity
    imo: str
    vessel_name: str
    vessel_type: str
    flag_country: str
    latitude: float
    longitude: float
    speed: float  # knots
    course: float  # degrees
    heading: float
    timestamp: datetime
    destination: str = ""
    eta: str = ""
    draught: float = 0.0


@dataclass
class TradeRouteActivity:
    """Aggregated shipping activity on a trade route."""

    route_name: str
    region_bbox: list[float]
    vessel_count: int
    tanker_count: int
    cargo_count: int
    military_count: int
    avg_speed: float
    period_start: datetime | None = None
    period_end: datetime | None = None
    anomalies: list[str] = field(default_factory=list)


# Strategic maritime chokepoints and regions
STRATEGIC_REGIONS: dict[str, dict[str, Any]] = {
    "strait_of_hormuz": {
        "name": "Strait of Hormuz",
        "bbox": [55.5, 25.5, 57.0, 27.0],
        "significance": "~20% of global oil transit",
    },
    "suez_canal": {
        "name": "Suez Canal",
        "bbox": [32.2, 29.8, 32.6, 31.3],
        "significance": "12% of global trade",
    },
    "taiwan_strait": {
        "name": "Taiwan Strait",
        "bbox": [119.0, 23.5, 121.0, 25.5],
        "significance": "Major semiconductor supply chain route",
    },
    "malacca_strait": {
        "name": "Strait of Malacca",
        "bbox": [99.5, 1.0, 104.5, 4.5],
        "significance": "25% of global shipping",
    },
    "bab_el_mandeb": {
        "name": "Bab el-Mandeb",
        "bbox": [42.5, 12.0, 44.0, 13.5],
        "significance": "Red Sea chokepoint, Yemen conflict zone",
    },
    "black_sea": {
        "name": "Black Sea",
        "bbox": [27.5, 40.5, 42.0, 47.5],
        "significance": "Ukraine grain exports, Russian naval base",
    },
    "south_china_sea": {
        "name": "South China Sea",
        "bbox": [105.0, 3.0, 121.0, 23.0],
        "significance": "Disputed territory, major trade route",
    },
}


class MarineTrafficClient:
    """Client for vessel tracking data.

    Supports MarineTraffic API (paid) and can also work with
    free AIS data sources for basic monitoring.
    """

    def __init__(self, config: MarineTrafficConfig | None = None):
        self.config = config or get_settings().marine_traffic
        self.session = requests.Session()

    def get_vessels_in_area(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        vessel_type: str | None = None,
    ) -> list[VesselPosition]:
        """Fetch current vessel positions within a bounding box.

        Requires MarineTraffic API key for live data.
        """
        if not self.config.api_key:
            logger.warning("MarineTraffic API key not configured")
            return []

        params: dict[str, Any] = {
            "v": "5",
            "apikey": self.config.api_key,
            "MINLAT": min_lat,
            "MAXLAT": max_lat,
            "MINLON": min_lon,
            "MAXLON": max_lon,
            "protocol": "json",
        }
        if vessel_type:
            params["shiptype"] = vessel_type

        try:
            url = f"{self.config.base_url}/exportvessel/v:5/{self.config.api_key}"
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("MarineTraffic API request failed")
            return []

        vessels = []
        for v in data if isinstance(data, list) else []:
            try:
                vessels.append(
                    VesselPosition(
                        mmsi=str(v.get("MMSI", "")),
                        imo=str(v.get("IMO", "")),
                        vessel_name=v.get("SHIPNAME", ""),
                        vessel_type=v.get("SHIPTYPE", ""),
                        flag_country=v.get("FLAG", ""),
                        latitude=float(v.get("LAT", 0)),
                        longitude=float(v.get("LON", 0)),
                        speed=float(v.get("SPEED", 0)) / 10.0,
                        course=float(v.get("COURSE", 0)),
                        heading=float(v.get("HEADING", 0)),
                        timestamp=datetime.utcnow(),
                        destination=v.get("DESTINATION", ""),
                    )
                )
            except (ValueError, KeyError):
                continue
        return vessels

    def monitor_strategic_region(self, region_key: str) -> TradeRouteActivity | None:
        """Monitor vessel activity in a strategic maritime region."""
        region = STRATEGIC_REGIONS.get(region_key)
        if not region:
            logger.error("Unknown strategic region: %s", region_key)
            return None

        bbox = region["bbox"]
        vessels = self.get_vessels_in_area(bbox[1], bbox[0], bbox[3], bbox[2])

        if not vessels:
            return TradeRouteActivity(
                route_name=region["name"],
                region_bbox=bbox,
                vessel_count=0,
                tanker_count=0,
                cargo_count=0,
                military_count=0,
                avg_speed=0,
            )

        tanker_types = {"tanker", "oil tanker", "chemical tanker", "lng carrier"}
        cargo_types = {"cargo", "container", "bulk carrier"}
        military_types = {"military", "naval", "warship"}

        tankers = sum(1 for v in vessels if v.vessel_type.lower() in tanker_types)
        cargo = sum(1 for v in vessels if v.vessel_type.lower() in cargo_types)
        military = sum(1 for v in vessels if v.vessel_type.lower() in military_types)
        avg_speed = sum(v.speed for v in vessels) / len(vessels)

        anomalies = []
        if military > 5:
            anomalies.append(f"Elevated military vessel count: {military}")
        if avg_speed < 2.0 and len(vessels) > 10:
            anomalies.append("Low average speed may indicate congestion or blockade")

        return TradeRouteActivity(
            route_name=region["name"],
            region_bbox=bbox,
            vessel_count=len(vessels),
            tanker_count=tankers,
            cargo_count=cargo,
            military_count=military,
            avg_speed=avg_speed,
            period_start=datetime.utcnow(),
            anomalies=anomalies,
        )
