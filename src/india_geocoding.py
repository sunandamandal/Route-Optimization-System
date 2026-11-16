"""
src/india_geocoding.py
India-Wide Location Search & OpenStreetMap Geocoding Module.

Features:
  1. Allows users to enter any Indian city, town, district, locality, or address.
  2. Resolves queries to precise latitude, longitude, and formatted Indian display names.
  3. Uses OpenStreetMap Nominatim with strict country restriction (countrycodes='in').
  4. Complies with OSM Nominatim usage policy:
     - Custom User-Agent and Referer headers
     - Rate-limiting (enforces minimum 1.0 second between consecutive live queries)
     - Attribution requirement: Data © OpenStreetMap contributors, ODbL 1.0
     - No aggressive autocomplete / as-you-type querying against the public service
  5. Validates that all resolved locations lie strictly within India's geographic boundaries.
  6. Two-tier caching (in-memory dict + persistent data/geocoding_cache.json).
  7. Built-in pre-seeded offline index of major Indian cities, towns, and hubs
     for zero-latency and resilient offline demonstration.
"""

import os
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "data" / "geocoding_cache.json"

# OSM Nominatim API endpoints & headers
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "EV-Route-Optimizer-India/2.0 (university.ai.ev.routing@gmail.com)"
REFERER = "https://nominatim.openstreetmap.org/"
OSM_ATTRIBUTION = "Data © OpenStreetMap contributors, ODbL 1.0 (https://osm.org/copyright)"

# India geographic bounding box (approximate coordinates)
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 37.5
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.5

# Pre-seeded offline index for major Indian cities, towns, and industrial hubs
# Provides immediate responses and works even if offline.
OFFLINE_INDIAN_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "kolkata": {
        "display_name": "Kolkata, West Bengal, India",
        "short_name": "Kolkata, WB",
        "lat": 22.5726,
        "lon": 88.3639,
        "state": "West Bengal",
        "elevation_m": 9.0
    },
    "howrah": {
        "display_name": "Howrah, West Bengal, India",
        "short_name": "Howrah, WB",
        "lat": 22.5736,
        "lon": 88.3103,
        "state": "West Bengal",
        "elevation_m": 12.0
    },
    "durgapur": {
        "display_name": "Durgapur, Paschim Bardhaman, West Bengal, India",
        "short_name": "Durgapur, WB",
        "lat": 23.5204,
        "lon": 87.3119,
        "state": "West Bengal",
        "elevation_m": 75.0
    },
    "asansol": {
        "display_name": "Asansol, Paschim Bardhaman, West Bengal, India",
        "short_name": "Asansol, WB",
        "lat": 23.6889,
        "lon": 86.9661,
        "state": "West Bengal",
        "elevation_m": 100.0
    },
    "siliguri": {
        "display_name": "Siliguri, Darjeeling, West Bengal, India",
        "short_name": "Siliguri, WB",
        "lat": 26.7271,
        "lon": 88.3953,
        "state": "West Bengal",
        "elevation_m": 122.0
    },
    "bardhaman": {
        "display_name": "Bardhaman, Purba Bardhaman, West Bengal, India",
        "short_name": "Bardhaman, WB",
        "lat": 23.2324,
        "lon": 87.8615,
        "state": "West Bengal",
        "elevation_m": 40.0
    },
    "kharagpur": {
        "display_name": "Kharagpur, Paschim Medinipur, West Bengal, India",
        "short_name": "Kharagpur, WB",
        "lat": 22.3460,
        "lon": 87.2320,
        "state": "West Bengal",
        "elevation_m": 61.0
    },
    "haldia": {
        "display_name": "Haldia, Purba Medinipur, West Bengal, India",
        "short_name": "Haldia, WB",
        "lat": 22.0620,
        "lon": 88.0700,
        "state": "West Bengal",
        "elevation_m": 8.0
    },
    "salt lake": {
        "display_name": "Salt Lake City, Bidhannagar, Kolkata, West Bengal, India",
        "short_name": "Salt Lake, Kolkata",
        "lat": 22.5868,
        "lon": 88.4178,
        "state": "West Bengal",
        "elevation_m": 11.0
    },
    "new town": {
        "display_name": "New Town, Rajarhat, Kolkata, West Bengal, India",
        "short_name": "New Town, Kolkata",
        "lat": 22.5850,
        "lon": 88.4700,
        "state": "West Bengal",
        "elevation_m": 10.0
    },
    "delhi": {
        "display_name": "Delhi, India",
        "short_name": "Delhi",
        "lat": 28.6665,
        "lon": 77.2167,
        "state": "Delhi",
        "elevation_m": 216.0
    },
    "new delhi": {
        "display_name": "New Delhi, Delhi, India",
        "short_name": "New Delhi",
        "lat": 28.6139,
        "lon": 77.2090,
        "state": "Delhi",
        "elevation_m": 215.0
    },
    "noida": {
        "display_name": "Noida, Gautam Buddha Nagar, Uttar Pradesh, India",
        "short_name": "Noida, UP",
        "lat": 28.5355,
        "lon": 77.3910,
        "state": "Uttar Pradesh",
        "elevation_m": 200.0
    },
    "gurgaon": {
        "display_name": "Gurugram (Gurgaon), Haryana, India",
        "short_name": "Gurugram, HR",
        "lat": 28.4595,
        "lon": 77.0266,
        "state": "Haryana",
        "elevation_m": 220.0
    },
    "mumbai": {
        "display_name": "Mumbai, Maharashtra, India",
        "short_name": "Mumbai, MH",
        "lat": 19.0550,
        "lon": 72.8753,
        "state": "Maharashtra",
        "elevation_m": 14.0
    },
    "pune": {
        "display_name": "Pune, Maharashtra, India",
        "short_name": "Pune, MH",
        "lat": 18.5204,
        "lon": 73.8567,
        "state": "Maharashtra",
        "elevation_m": 560.0
    },
    "navi mumbai": {
        "display_name": "Navi Mumbai, Thane, Maharashtra, India",
        "short_name": "Navi Mumbai, MH",
        "lat": 19.0330,
        "lon": 73.0297,
        "state": "Maharashtra",
        "elevation_m": 10.0
    },
    "bengaluru": {
        "display_name": "Bengaluru, Karnataka, India",
        "short_name": "Bengaluru, KA",
        "lat": 12.9768,
        "lon": 77.5946,
        "state": "Karnataka",
        "elevation_m": 920.0
    },
    "electronic city": {
        "display_name": "Electronic City, Bengaluru, Karnataka, India",
        "short_name": "Electronic City, BLR",
        "lat": 12.8399,
        "lon": 77.6770,
        "state": "Karnataka",
        "elevation_m": 890.0
    },
    "whitefield": {
        "display_name": "Whitefield, Bengaluru, Karnataka, India",
        "short_name": "Whitefield, BLR",
        "lat": 12.9698,
        "lon": 77.7500,
        "state": "Karnataka",
        "elevation_m": 895.0
    },
    "mysuru": {
        "display_name": "Mysuru, Karnataka, India",
        "short_name": "Mysuru, KA",
        "lat": 12.2958,
        "lon": 76.6394,
        "state": "Karnataka",
        "elevation_m": 770.0
    },
    "chennai": {
        "display_name": "Chennai, Tamil Nadu, India",
        "short_name": "Chennai, TN",
        "lat": 13.0827,
        "lon": 80.2707,
        "state": "Tamil Nadu",
        "elevation_m": 6.0
    },
    "hyderabad": {
        "display_name": "Hyderabad, Telangana, India",
        "short_name": "Hyderabad, TS",
        "lat": 17.3850,
        "lon": 78.4867,
        "state": "Telangana",
        "elevation_m": 505.0
    },
    "ahmedabad": {
        "display_name": "Ahmedabad, Gujarat, India",
        "short_name": "Ahmedabad, GJ",
        "lat": 23.0225,
        "lon": 72.5714,
        "state": "Gujarat",
        "elevation_m": 53.0
    },
    "jaipur": {
        "display_name": "Jaipur, Rajasthan, India",
        "short_name": "Jaipur, RJ",
        "lat": 26.9124,
        "lon": 75.7873,
        "state": "Rajasthan",
        "elevation_m": 431.0
    },
    "lucknow": {
        "display_name": "Lucknow, Uttar Pradesh, India",
        "short_name": "Lucknow, UP",
        "lat": 26.8467,
        "lon": 80.9462,
        "state": "Uttar Pradesh",
        "elevation_m": 123.0
    },
    "patna": {
        "display_name": "Patna, Bihar, India",
        "short_name": "Patna, BR",
        "lat": 25.5941,
        "lon": 85.1376,
        "state": "Bihar",
        "elevation_m": 53.0
    },
    "ranchi": {
        "display_name": "Ranchi, Jharkhand, India",
        "short_name": "Ranchi, JH",
        "lat": 23.3441,
        "lon": 85.3096,
        "state": "Jharkhand",
        "elevation_m": 651.0
    },
    "bhubaneswar": {
        "display_name": "Bhubaneswar, Odisha, India",
        "short_name": "Bhubaneswar, OD",
        "lat": 20.2961,
        "lon": 85.8245,
        "state": "Odisha",
        "elevation_m": 45.0
    },
    "chandigarh": {
        "display_name": "Chandigarh, India",
        "short_name": "Chandigarh",
        "lat": 30.7333,
        "lon": 76.7794,
        "state": "Chandigarh",
        "elevation_m": 321.0
    }
}


@dataclass
class GeocodedLocation:
    """Represents a validated Indian geographic location."""
    query: str
    display_name: str
    short_name: str
    lat: float
    lon: float
    state: str
    country: str
    elevation_m: float
    source: str          # "offline_index", "cache", or "nominatim"
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndiaGeocoder:
    """
    OpenStreetMap-compatible geocoding service for Indian locations.
    Enforces rate limits, validates bounds, caches results, and falls back to offline data.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = Path(cache_path) if cache_path else CACHE_FILE
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._last_request_time: float = 0.0
        self.attribution: str = OSM_ATTRIBUTION

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads cached geocoding results from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persists the geocoding cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Best effort persistence

    def _rate_limit(self):
        """Strictly respects Nominatim's 1-request-per-second policy."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _clean_query(query: str) -> str:
        """Normalizes user search text."""
        cleaned = re.sub(r"\s+", " ", query.strip())
        return cleaned

    @staticmethod
    def is_within_india(lat: float, lon: float) -> bool:
        """Validates that coordinates lie within India's territorial bounding box."""
        return (INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX) and (INDIA_LON_MIN <= lon <= INDIA_LON_MAX)

    def geocode(self, query: str) -> GeocodedLocation:
        """
        Geocodes any Indian location query into a validated GeocodedLocation.

        Parameters:
            query (str): Location name, city, town, locality, or address.

        Returns:
            GeocodedLocation: Validated location object with latitude and longitude.
        """
        cleaned_query = self._clean_query(query)
        if not cleaned_query:
            return GeocodedLocation(
                query=query,
                display_name="",
                short_name="",
                lat=0.0,
                lon=0.0,
                state="",
                country="",
                elevation_m=0.0,
                source="error",
                is_valid=False,
                error_message="Location query cannot be empty. Please enter an Indian location."
            )

        query_key = cleaned_query.lower()

        # 1. Check in-memory / persistent disk cache
        if query_key in self._cache:
            data = self._cache[query_key]
            return GeocodedLocation(**data)

        # 2. Check offline pre-seeded database
        # Exact match or substring match (e.g. "kolkata", "kolkata, west bengal", "howrah station")
        for key, info in OFFLINE_INDIAN_LOCATIONS.items():
            if key == query_key or key in query_key or query_key in info["display_name"].lower():
                loc = GeocodedLocation(
                    query=cleaned_query,
                    display_name=info["display_name"],
                    short_name=info["short_name"],
                    lat=float(info["lat"]),
                    lon=float(info["lon"]),
                    state=info["state"],
                    country="India",
                    elevation_m=float(info["elevation_m"]),
                    source="offline_index",
                    is_valid=True
                )
                self._cache[query_key] = loc.to_dict()
                self._save_cache()
                return loc

        # 3. Live query to OpenStreetMap Nominatim
        self._rate_limit()

        params = {
            "q": cleaned_query,
            "format": "json",
            "limit": 1,
            "countrycodes": "in",
            "addressdetails": 1
        }
        headers = {
            "User-Agent": USER_AGENT,
            "Referer": REFERER
        }

        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=6.0)
            if resp.status_code == 200:
                results = resp.json()
                if not results:
                    return GeocodedLocation(
                        query=cleaned_query,
                        display_name="",
                        short_name="",
                        lat=0.0,
                        lon=0.0,
                        state="",
                        country="",
                        elevation_m=0.0,
                        source="nominatim",
                        is_valid=False,
                        error_message=(
                            f"Location '{cleaned_query}' could not be found in India. "
                            "Please check the spelling or specify the district/state (e.g., 'Durgapur, West Bengal')."
                        )
                    )

                top_result = results[0]
                lat = float(top_result["lat"])
                lon = float(top_result["lon"])
                address = top_result.get("address", {})

                country = address.get("country", "India")
                country_code = address.get("country_code", "in").lower()
                state = address.get("state", address.get("region", "India"))

                # Strict validation: Location must be in India
                if country_code != "in" or not self.is_within_india(lat, lon):
                    return GeocodedLocation(
                        query=cleaned_query,
                        display_name=top_result.get("display_name", ""),
                        short_name="",
                        lat=lat,
                        lon=lon,
                        state=state,
                        country=country,
                        elevation_m=0.0,
                        source="nominatim",
                        is_valid=False,
                        error_message=(
                            f"Resolved location '{cleaned_query}' is outside India ({country}). "
                            "Only Indian locations are supported by this system."
                        )
                    )

                display_name = top_result.get("display_name", cleaned_query)
                # Build a concise readable short name
                parts = [p.strip() for p in display_name.split(",")]
                short_name = ", ".join(parts[:2]) if len(parts) >= 2 else display_name

                # Estimate elevation baseline from state/terrain if not provided
                elevation_m = self._estimate_elevation(lat, lon, state)

                loc = GeocodedLocation(
                    query=cleaned_query,
                    display_name=display_name,
                    short_name=short_name,
                    lat=round(lat, 6),
                    lon=round(lon, 6),
                    state=state,
                    country="India",
                    elevation_m=round(elevation_m, 1),
                    source="nominatim",
                    is_valid=True
                )

                # Cache validated result
                self._cache[query_key] = loc.to_dict()
                self._save_cache()
                return loc

            else:
                return self._fallback_or_error(
                    cleaned_query,
                    f"Nominatim geocoding service returned HTTP status {resp.status_code}."
                )

        except requests.exceptions.Timeout:
            return self._fallback_or_error(
                cleaned_query,
                "Geocoding request timed out. Please check your network connection."
            )
        except requests.exceptions.RequestException as e:
            return self._fallback_or_error(
                cleaned_query,
                f"Network error connecting to OpenStreetMap geocoding: {e}"
            )
        except Exception as e:
            return self._fallback_or_error(
                cleaned_query,
                f"Unexpected geocoding error: {e}"
            )

    def _fallback_or_error(self, query: str, reason: str) -> GeocodedLocation:
        """Attempts partial match in offline database before failing."""
        query_lower = query.lower()
        for key, info in OFFLINE_INDIAN_LOCATIONS.items():
            if key in query_lower or query_lower in key:
                return GeocodedLocation(
                    query=query,
                    display_name=info["display_name"],
                    short_name=info["short_name"],
                    lat=float(info["lat"]),
                    lon=float(info["lon"]),
                    state=info["state"],
                    country="India",
                    elevation_m=float(info["elevation_m"]),
                    source="offline_fallback",
                    is_valid=True
                )

        return GeocodedLocation(
            query=query,
            display_name="",
            short_name="",
            lat=0.0,
            lon=0.0,
            state="",
            country="",
            elevation_m=0.0,
            source="error",
            is_valid=False,
            error_message=reason
        )

    @staticmethod
    def _estimate_elevation(lat: float, lon: float, state: str) -> float:
        """Approximates realistic elevation based on Indian geographical regions."""
        st = state.lower()
        if any(h in st for h in ["ladakh", "jammu", "kashmir", "himachal", "uttarakhand", "sikkim"]):
            return 1200.0
        elif any(p in st for p in ["karnataka", "telangana", "maharashtra"]):
            return 550.0
        elif any(p in st for p in ["madhya pradesh", "jharkhand", "chhattisgarh"]):
            return 350.0
        elif any(c in st for c in ["delhi", "haryana", "punjab", "uttar pradesh"]):
            return 210.0
        elif any(c in st for c in ["west bengal", "odisha", "tamil nadu", "kerala", "andhra pradesh", "goa"]):
            # Coastal / delta plains
            return 20.0
        return 100.0


# Module-level singleton
_GEOCODER_INSTANCE: Optional[IndiaGeocoder] = None


def get_india_geocoder() -> IndiaGeocoder:
    """Returns the singleton instance of IndiaGeocoder."""
    global _GEOCODER_INSTANCE
    if _GEOCODER_INSTANCE is None:
        _GEOCODER_INSTANCE = IndiaGeocoder()
    return _GEOCODER_INSTANCE


def geocode_indian_location(query: str) -> GeocodedLocation:
    """Convenience helper to geocode any Indian location query."""
    return get_india_geocoder().geocode(query)
