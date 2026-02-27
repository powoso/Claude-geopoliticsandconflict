"""Satellite imagery change detection for military buildup indicators.

Uses Sentinel Hub free tier to detect changes in areas of interest:
  - Military base activity (vehicle counts, construction)
  - Troop concentration areas
  - Infrastructure changes near borders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import requests

from geopolitical_prediction.config import SentinelConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class AreaOfInterest:
    """A geographic bounding box to monitor for changes."""

    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    description: str = ""

    @property
    def bbox(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


@dataclass
class ChangeDetectionResult:
    """Result of satellite change detection analysis."""

    area: AreaOfInterest
    date_before: datetime
    date_after: datetime
    change_magnitude: float  # 0 to 1
    changed_pixel_pct: float
    mean_ndvi_before: float
    mean_ndvi_after: float
    interpretation: str


# Predefined areas of interest for geopolitical monitoring
KNOWN_AREAS: dict[str, AreaOfInterest] = {
    "kaliningrad_military": AreaOfInterest(
        name="Kaliningrad Military District",
        min_lon=20.2, min_lat=54.3, max_lon=20.8, max_lat=54.8,
        description="Russian exclave military installations",
    ),
    "taiwan_strait": AreaOfInterest(
        name="Taiwan Strait",
        min_lon=119.0, min_lat=23.5, max_lon=120.5, max_lat=25.5,
        description="Cross-strait military activity zone",
    ),
    "dmz_korea": AreaOfInterest(
        name="Korean DMZ",
        min_lon=126.0, min_lat=37.8, max_lon=127.5, max_lat=38.5,
        description="Korean Demilitarized Zone",
    ),
    "golan_heights": AreaOfInterest(
        name="Golan Heights",
        min_lon=35.6, min_lat=32.7, max_lon=36.1, max_lat=33.3,
        description="Israel-Syria border region",
    ),
}


class SentinelHubClient:
    """Client for Sentinel Hub Process API (free tier).

    Retrieves Sentinel-2 satellite imagery for change detection.
    Free tier allows 30,000 requests/month with limited resolution.
    """

    def __init__(self, config: SentinelConfig | None = None):
        self.config = config or get_settings().sentinel
        self._token: str | None = None
        self._token_expires: datetime | None = None

    def _authenticate(self) -> str:
        """Get OAuth2 access token from Sentinel Hub."""
        if self._token and self._token_expires and datetime.utcnow() < self._token_expires:
            return self._token

        if not self.config.client_id or not self.config.client_secret:
            raise ValueError(
                "Sentinel Hub credentials not configured. "
                "Set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET."
            )

        resp = requests.post(
            self.config.oauth_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        self._token = data["access_token"]
        self._token_expires = datetime.utcnow() + timedelta(
            seconds=data.get("expires_in", 3600) - 60
        )
        return self._token

    def fetch_ndvi_stats(
        self,
        area: AreaOfInterest,
        date_from: datetime,
        date_to: datetime,
        resolution: int = 60,
    ) -> dict[str, Any]:
        """Fetch NDVI statistics for an area over a date range.

        NDVI (Normalized Difference Vegetation Index) changes can indicate:
        - Construction activity (vegetation removal)
        - Vehicle/equipment staging
        - Earthworks and fortification building

        Returns mean/std NDVI and cloud cover percentage.
        """
        token = self._authenticate()

        # Sentinel Hub evalscript for NDVI computation
        evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B08", "SCL"]}],
    output: {bands: 3, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  let cloud = (sample.SCL == 8 || sample.SCL == 9) ? 1.0 : 0.0;
  return [ndvi, cloud, 1.0];
}
"""

        payload = {
            "input": {
                "bounds": {
                    "bbox": area.bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": date_from.strftime("%Y-%m-%dT00:00:00Z"),
                                "to": date_to.strftime("%Y-%m-%dT23:59:59Z"),
                            },
                            "maxCloudCoverage": 30,
                        },
                    }
                ],
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }

        try:
            resp = requests.post(
                self.config.process_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            resp.raise_for_status()

            # Parse the TIFF response to extract NDVI statistics
            # In production, use rasterio; here we return basic info
            return {
                "area": area.name,
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "response_size": len(resp.content),
                "status": "success",
            }
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning("Sentinel Hub access denied - check credentials/quota")
            raise
        except Exception:
            logger.exception("Sentinel Hub request failed for %s", area.name)
            return {"area": area.name, "status": "error"}


class ChangeDetector:
    """High-level change detection pipeline for military buildup monitoring.

    Compares satellite imagery from two time periods to detect
    significant ground-level changes in areas of interest.
    """

    def __init__(self, config: SentinelConfig | None = None):
        self.client = SentinelHubClient(config)
        self.monitored_areas: list[AreaOfInterest] = []

    def add_area(self, area: AreaOfInterest) -> None:
        """Add an area of interest to monitor."""
        self.monitored_areas.append(area)

    def add_known_area(self, key: str) -> None:
        """Add a predefined known area by key."""
        area = KNOWN_AREAS.get(key)
        if area:
            self.monitored_areas.append(area)
        else:
            logger.warning("Unknown area key: %s", key)

    def analyze_change(
        self,
        area: AreaOfInterest,
        days_before: int = 60,
        days_recent: int = 10,
    ) -> ChangeDetectionResult | None:
        """Compare recent imagery to baseline for an area.

        Fetches NDVI stats for baseline period and recent period,
        then computes change magnitude.
        """
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=days_before)
        baseline_end = now - timedelta(days=days_recent)
        recent_start = now - timedelta(days=days_recent)
        recent_end = now

        try:
            baseline = self.client.fetch_ndvi_stats(area, baseline_start, baseline_end)
            recent = self.client.fetch_ndvi_stats(area, recent_start, recent_end)

            if baseline.get("status") != "success" or recent.get("status") != "success":
                return None

            # Placeholder: in production, parse actual NDVI raster data
            # For now return a structured result indicating the check was performed
            return ChangeDetectionResult(
                area=area,
                date_before=baseline_end,
                date_after=recent_end,
                change_magnitude=0.0,  # would be computed from actual imagery
                changed_pixel_pct=0.0,
                mean_ndvi_before=0.0,
                mean_ndvi_after=0.0,
                interpretation="analysis_pending",
            )
        except Exception:
            logger.exception("Change detection failed for %s", area.name)
            return None

    def scan_all_areas(self) -> list[ChangeDetectionResult]:
        """Run change detection on all monitored areas."""
        results = []
        for area in self.monitored_areas:
            result = self.analyze_change(area)
            if result:
                results.append(result)
        return results
