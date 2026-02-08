"""
Location RAG System with LangChain Retriever and GeoPandas Geofencing
======================================================================
Uses GeoPandas for spatial queries and polygon geometry, with a LangChain
custom retriever backed by FAISS for semantic search. Provides rich,
contextual location descriptions for Ranni's GPS coordinates.

Dependencies:
    pip install geopandas langchain langchain-community langchain-huggingface
    pip install sentence-transformers faiss-cpu

GeoPandas internally uses Shapely geometry objects and adds a powerful
spatial indexing layer (R-tree via Shapely 2.0), making operations
like "which polygon contains this point?" or "find nearest polygon"
significantly faster at scale than iterating manually.

LangChain's BaseRetriever abstraction lets us wrap our spatial +
metadata search as a proper retriever that can plug into any LangChain
chain or agent down the road.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, mapping
import pandas as pd

# LangChain imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Pydantic v2 compat for the retriever
from pydantic import Field as PydanticField

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LocationZone:
    """
    A geographic zone defined by a convex polygon with rich metadata.
    Unchanged from the original interface so voice_agent.py / api.py
    continue to work without modification.
    """
    id: str
    name: str
    description: str
    polygon_coords: List[Tuple[float, float]]  # (lon, lat) tuples
    zone_type: str = "general"
    safety_level: str = "safe"
    typical_activities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    time_context: Dict[str, str] = field(default_factory=dict)

    def to_shapely(self) -> Polygon:
        """Convert to Shapely Polygon (kept for backward compat)."""
        return Polygon(self.polygon_coords)

    def contains_point(self, lon: float, lat: float) -> bool:
        point = Point(lon, lat)
        return self.to_shapely().contains(point)

    def distance_to_point(self, lon: float, lat: float) -> float:
        point = Point(lon, lat)
        polygon = self.to_shapely()
        if polygon.contains(point):
            return 0.0
        return point.distance(polygon)


@dataclass
class LocationContext:
    """Rich context about a location query result (unchanged interface)."""
    zone: Optional[LocationZone]
    is_inside_zone: bool
    distance_m: float
    semantic_matches: List[Dict[str, Any]]
    narrative: str
    safety_alert: Optional[str] = None
    coordinates: Tuple[float, float] = (0.0, 0.0)
    timestamp: str = ""


# ============================================================================
# LANGCHAIN CUSTOM RETRIEVER — wraps spatial + semantic search
# ============================================================================

class SpatialSemanticRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that combines:
      1. FAISS vector similarity on zone description embeddings
      2. GeoPandas spatial queries (containment, nearest) on polygons

    This means you can later drop it into any LangChain chain:
        chain = create_retrieval_chain(retriever, ...)
    """

    vectorstore: Any = PydanticField(default=None, exclude=True)
    zone_lookup: Dict[str, "LocationZone"] = PydanticField(
        default_factory=dict, exclude=True
    )
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        LangChain retriever interface — called by .invoke() / .get_relevant_documents().
        Returns Document objects with zone metadata.
        """
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        docs = []
        for doc, score in results:
            zone_id = doc.metadata.get("zone_id", "")
            zone = self.zone_lookup.get(zone_id)
            # Attach the zone object and score as metadata for downstream use
            enriched = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "score": float(score),
                    "zone": zone,
                },
            )
            docs.append(enriched)
        return docs


# ============================================================================
# LOCATION RAG DATABASE — GeoPandas + LangChain FAISS
# ============================================================================

class LocationRAG:
    """
    RAG system using GeoPandas for spatial queries and LangChain FAISS
    for semantic search.
    """

    METERS_PER_DEGREE = 111000  # approximate at mid-latitudes

    def __init__(self, db_path: str = "./ranni_location_db"):
        self.db_path = db_path
        self.zones: Dict[str, LocationZone] = {}

        # GeoPandas GeoDataFrame — spatial index is built automatically
        self._gdf: Optional[gpd.GeoDataFrame] = None

        # LangChain embeddings + FAISS vectorstore
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self._vectorstore: Optional[FAISS] = None

        # The LangChain retriever (created after first zone is added)
        self.retriever: Optional[SpatialSemanticRetriever] = None

        # Load persisted zones if available
        self._load_zones_from_disk()

        logger.info(f"LocationRAG (GeoPandas+LangChain) initialized — DB at {db_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_geodataframe(self) -> None:
        """Rebuild the GeoDataFrame from self.zones dict."""
        if not self.zones:
            self._gdf = None
            return

        rows = []
        for z in self.zones.values():
            rows.append({
                "zone_id": z.id,
                "name": z.name,
                "zone_type": z.zone_type,
                "safety_level": z.safety_level,
                "geometry": Polygon(z.polygon_coords),
            })

        self._gdf = gpd.GeoDataFrame(rows, geometry="geometry")
        # Build spatial index (happens automatically in modern GeoPandas)
        # but we call .sindex explicitly to ensure it's ready
        _ = self._gdf.sindex
        logger.debug(f"GeoDataFrame rebuilt with {len(self._gdf)} zones")

    def _rebuild_vectorstore(self) -> None:
        """Rebuild the FAISS vectorstore from self.zones dict."""
        if not self.zones:
            self._vectorstore = None
            self.retriever = None
            return

        docs: List[Document] = []
        for z in self.zones.values():
            text = self._create_embedding_text(z)
            doc = Document(
                page_content=text,
                metadata={
                    "zone_id": z.id,
                    "name": z.name,
                    "zone_type": z.zone_type,
                    "safety_level": z.safety_level,
                    "tags": ",".join(z.tags),
                },
            )
            docs.append(doc)

        self._vectorstore = FAISS.from_documents(docs, self._embeddings)

        # Wire up the LangChain retriever
        self.retriever = SpatialSemanticRetriever(
            vectorstore=self._vectorstore,
            zone_lookup=self.zones,
        )
        logger.debug("FAISS vectorstore rebuilt")

    @staticmethod
    def _create_embedding_text(zone: LocationZone) -> str:
        """Create rich text for semantic embedding (same logic as original)."""
        parts = [
            f"Location: {zone.name}",
            f"Description: {zone.description}",
            f"Type: {zone.zone_type}",
            f"Safety: {zone.safety_level}",
        ]
        if zone.typical_activities:
            parts.append(f"Activities: {', '.join(zone.typical_activities)}")
        if zone.tags:
            parts.append(f"Tags: {', '.join(zone.tags)}")
        for time_period, context in zone.time_context.items():
            parts.append(f"At {time_period}: {context}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_zone(self, zone: LocationZone) -> None:
        """Add a location zone (updates both spatial index and vectorstore)."""
        self.zones[zone.id] = zone
        # Rebuild indexes — cheap for a dozen zones; for hundreds you could
        # do incremental inserts on the FAISS side.
        self._rebuild_geodataframe()
        self._rebuild_vectorstore()
        logger.info(f"Added zone: {zone.name} ({zone.id})")

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self.zones:
            del self.zones[zone_id]
            self._rebuild_geodataframe()
            self._rebuild_vectorstore()
            return True
        return False

    def find_containing_zone(self, lon: float, lat: float) -> Optional[LocationZone]:
        """
        Find which zone contains the given point using GeoPandas spatial index.
        Much faster than iterating when you have many zones.
        """
        if self._gdf is None or self._gdf.empty:
            return None

        point = Point(lon, lat)

        # Use spatial index for fast candidate filtering, then precise check
        possible_idx = list(self._gdf.sindex.query(point, predicate="intersects"))

        for idx in possible_idx:
            row = self._gdf.iloc[idx]
            if row.geometry.contains(point):
                return self.zones.get(row["zone_id"])

        return None

    def find_nearest_zone(
        self, lon: float, lat: float
    ) -> Tuple[Optional[LocationZone], float]:
        """
        Find the nearest zone using GeoPandas distance operations.
        Returns (zone, distance_in_meters).
        """
        if self._gdf is None or self._gdf.empty:
            return None, float("inf")

        point = Point(lon, lat)

        # Check containment first (distance = 0)
        containing = self.find_containing_zone(lon, lat)
        if containing:
            return containing, 0.0

        # Vectorised distance calculation across all polygons
        distances = self._gdf.geometry.distance(point)
        nearest_idx = distances.idxmin()
        nearest_row = self._gdf.iloc[nearest_idx]
        distance_deg = distances.iloc[nearest_idx]

        zone = self.zones.get(nearest_row["zone_id"])
        return zone, distance_deg * self.METERS_PER_DEGREE

    def semantic_search(
        self,
        query: str,
        n_results: int = 3,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on zone descriptions via LangChain FAISS.
        Returns the same list-of-dicts format as the original.
        """
        if self._vectorstore is None:
            return []

        # LangChain FAISS supports filter dicts on metadata
        filter_dict = None
        if filter_type:
            filter_dict = {"zone_type": filter_type}

        results = self._vectorstore.similarity_search_with_score(
            query, k=n_results, filter=filter_dict
        )

        matches = []
        for doc, score in results:
            zone_id = doc.metadata.get("zone_id", "")
            zone = self.zones.get(zone_id)
            if zone:
                matches.append({
                    "zone": zone,
                    "score": float(score),
                    "document": doc.page_content,
                })
        return matches

    def get_location_context(
        self,
        lon: float,
        lat: float,
        semantic_query: Optional[str] = None,
    ) -> LocationContext:
        """
        Get rich context combining spatial and semantic search.
        Interface identical to the original.
        """
        timestamp = datetime.now().isoformat()
        current_hour = datetime.now().hour

        containing_zone = self.find_containing_zone(lon, lat)
        if containing_zone:
            zone = containing_zone
            is_inside = True
            distance_m = 0.0
        else:
            zone, distance_m = self.find_nearest_zone(lon, lat)
            is_inside = False

        query = semantic_query or "cat location whereabouts position"
        semantic_matches = self.semantic_search(query, n_results=3)

        narrative = self._generate_narrative(
            zone=zone,
            is_inside=is_inside,
            distance_m=distance_m,
            hour=current_hour,
            lon=lon,
            lat=lat,
        )

        safety_alert = None
        if zone and zone.safety_level == "danger":
            safety_alert = f"WARNING: {zone.name} is marked as a dangerous area!"
        elif zone and zone.safety_level == "caution":
            safety_alert = f"Caution advised: {zone.name} requires attention."

        return LocationContext(
            zone=zone,
            is_inside_zone=is_inside,
            distance_m=distance_m,
            semantic_matches=semantic_matches,
            narrative=narrative,
            safety_alert=safety_alert,
            coordinates=(lon, lat),
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    def _generate_narrative(
        self,
        zone: Optional[LocationZone],
        is_inside: bool,
        distance_m: float,
        hour: int,
        lon: float,
        lat: float,
    ) -> str:
        if zone is None:
            return (
                f"The feline wanders in uncharted territory, beyond the boundaries "
                f"of mapped lands. Coordinates: {lat:.6f}, {lon:.6f}"
            )

        if 6 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 17:
            time_period = "afternoon"
        elif 17 <= hour < 21:
            time_period = "evening"
        else:
            time_period = "night"

        time_context = zone.time_context.get(time_period, "")

        if is_inside:
            narrative = f"Ranni dwells within {zone.name} - {zone.description}."
            if zone.typical_activities:
                narrative += f" She is likely {zone.typical_activities[0]}."
            if time_context:
                narrative += f" {time_context}"
        else:
            direction = self._get_direction_to_zone(lon, lat, zone)
            if distance_m < 50:
                proximity = "mere paces from"
            elif distance_m < 150:
                proximity = "near"
            elif distance_m < 500:
                proximity = "some distance from"
            else:
                proximity = "far from"
            narrative = (
                f"Ranni prowls {proximity} {zone.name}, approximately "
                f"{int(distance_m)} meters to the {direction}. {zone.description}."
            )

        if zone.safety_level == "danger":
            narrative += " DANGER: This area poses risks to her safety!"
        elif zone.safety_level == "caution":
            narrative += " Caution is advised in this area."

        return narrative

    def _get_direction_to_zone(
        self, lon: float, lat: float, zone: LocationZone
    ) -> str:
        polygon = zone.to_shapely()
        centroid = polygon.centroid
        delta_lon = centroid.x - lon
        delta_lat = centroid.y - lat
        if abs(delta_lat) > abs(delta_lon):
            return "north" if delta_lat > 0 else "south"
        return "east" if delta_lon > 0 else "west"

    # ------------------------------------------------------------------
    # Persistence (JSON — same format as original)
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Save zone data to disk. FAISS index is rebuilt on load."""
        zones_data = {zid: asdict(z) for zid, z in self.zones.items()}
        os.makedirs(self.db_path, exist_ok=True)
        zones_path = os.path.join(self.db_path, "zones.json")
        with open(zones_path, "w") as f:
            json.dump(zones_data, f, indent=2)

        # Optionally persist the FAISS index for faster cold starts
        if self._vectorstore is not None:
            faiss_dir = os.path.join(self.db_path, "faiss_index")
            self._vectorstore.save_local(faiss_dir)

        logger.info("Zone data persisted")

    def _load_zones_from_disk(self) -> None:
        zones_path = os.path.join(self.db_path, "zones.json")
        if not os.path.exists(zones_path):
            return

        try:
            with open(zones_path, "r") as f:
                zones_data = json.load(f)

            for zone_id, data in zones_data.items():
                zone = LocationZone(
                    id=data.get("id", zone_id),
                    name=data["name"],
                    description=data["description"],
                    polygon_coords=[tuple(c) for c in data["polygon_coords"]],
                    zone_type=data.get("zone_type", "general"),
                    safety_level=data.get("safety_level", "safe"),
                    typical_activities=data.get("typical_activities", []),
                    tags=data.get("tags", []),
                    time_context=data.get("time_context", {}),
                )
                self.zones[zone.id] = zone

            # Rebuild indexes once after loading all zones
            self._rebuild_geodataframe()

            # Try loading persisted FAISS, fall back to rebuilding
            faiss_dir = os.path.join(self.db_path, "faiss_index")
            if os.path.exists(faiss_dir):
                try:
                    self._vectorstore = FAISS.load_local(
                        faiss_dir,
                        self._embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    self.retriever = SpatialSemanticRetriever(
                        vectorstore=self._vectorstore,
                        zone_lookup=self.zones,
                    )
                    logger.info("Loaded persisted FAISS index")
                except Exception:
                    logger.warning("Could not load FAISS index, rebuilding")
                    self._rebuild_vectorstore()
            else:
                self._rebuild_vectorstore()

            logger.info(f"Loaded {len(self.zones)} zones from disk")
        except Exception as e:
            logger.warning(f"Could not load zones from disk: {e}")

    def load_zones_from_json(self, json_path: str) -> int:
        with open(json_path, "r") as f:
            zones_data = json.load(f)

        count = 0
        for zone_id, data in zones_data.items():
            zone = LocationZone(
                id=data.get("id", zone_id),
                name=data["name"],
                description=data["description"],
                polygon_coords=[tuple(c) for c in data["polygon_coords"]],
                zone_type=data.get("zone_type", "general"),
                safety_level=data.get("safety_level", "safe"),
                typical_activities=data.get("typical_activities", []),
                tags=data.get("tags", []),
                time_context=data.get("time_context", {}),
            )
            self.zones[zone.id] = zone
            count += 1

        # Rebuild once after all zones loaded (more efficient)
        self._rebuild_geodataframe()
        self._rebuild_vectorstore()
        return count


# ============================================================================
# EXAMPLE ZONE CONFIGURATION
# ============================================================================

def create_example_zones() -> List[LocationZone]:
    """
    Create example zones - UPDATE THESE WITH YOUR ACTUAL COORDINATES.

    To get polygon coordinates:
    1. Go to Google Maps
    2. Right-click to get lat/lon of each corner
    3. Create polygon as list of (longitude, latitude) tuples
    4. Make sure to close the polygon (first point = last point)

    Note: Coordinates are (longitude, latitude) for Shapely/GeoPandas compatibility!
    """

    zones = [
        # EXAMPLE ZONE
        LocationZone(
            id="backyard",
            name="The Rear Gardens",
            description="the verdant hunting grounds behind the estate",
            polygon_coords=[
                (-122.001, 37.000),  # (longitude, latitude) - SW corner
                (-122.001, 37.001),  # NW corner
                (-121.999, 37.001),  # NE corner
                (-121.999, 37.000),  # SE corner
                (-122.001, 37.000),  # Close the polygon
            ],
            zone_type="yard",
            safety_level="safe",  # safe, caution, danger
            typical_activities=[
                "stalking unsuspecting birds",
                "rolling in patches of sunlight",
            ],
            tags=["outdoor", "safe", "hunting"],
            time_context={
                "morning": "Dew glistens on the grass.",
                "night": "Prime hunting conditions.",
            }
        )
    ]

    return zones


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_location_rag: Optional[LocationRAG] = None


def get_location_rag() -> LocationRAG:
    """Get or create the global LocationRAG instance."""
    global _location_rag

    if _location_rag is None:
        _location_rag = LocationRAG(db_path="./ranni_location_db")

        if len(_location_rag.zones) == 0:
            logger.info("No zones found, loading example zones...")
            example_zones = create_example_zones()
            for zone in example_zones:
                _location_rag.add_zone(zone)
            _location_rag.persist()
            logger.info(f"Loaded {len(example_zones)} example zones")

    return _location_rag


# ============================================================================
# CLI TOOLS
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Location RAG System (GeoPandas + LangChain)")
    print("=" * 50)

    rag = get_location_rag()
    print(f"Loaded {len(rag.zones)} zones")

    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("\nConfigured Zones:")
            for zone_id, zone in rag.zones.items():
                print(f"\n  {zone.name} ({zone_id})")
                print(f"    Type: {zone.zone_type}, Safety: {zone.safety_level}")
                print(f"    Description: {zone.description[:60]}...")

        elif sys.argv[1] == "--test" and len(sys.argv) >= 4:
            lon = float(sys.argv[2])
            lat = float(sys.argv[3])
            print(f"\nTesting coordinates: ({lon}, {lat})")
            context = rag.get_location_context(lon, lat)
            print(f"\nResult:")
            print(f"  Zone: {context.zone.name if context.zone else 'Unknown'}")
            print(f"  Inside: {context.is_inside_zone}")
            print(f"  Distance: {context.distance_m:.1f}m")
            print(f"\nNarrative: {context.narrative}")
            if context.safety_alert:
                print(f"\n⚠️  {context.safety_alert}")

        elif sys.argv[1] == "--search":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "safe place to rest"
            print(f"\nSemantic search: '{query}'")
            results = rag.semantic_search(query, n_results=3)
            for i, match in enumerate(results, 1):
                print(f"\n  {i}. {match['zone'].name}")
                print(f"     Score: {match['score']:.4f}")
                print(f"     {match['zone'].description[:60]}...")

        elif sys.argv[1] == "--retriever":
            # Demo: use the LangChain retriever directly
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "dangerous area"
            print(f"\nLangChain retriever query: '{query}'")
            if rag.retriever:
                docs = rag.retriever.invoke(query)
                for i, doc in enumerate(docs, 1):
                    z = doc.metadata.get("zone")
                    print(f"\n  {i}. {z.name if z else '?'} (score: {doc.metadata['score']:.4f})")
                    print(f"     {doc.page_content[:80]}...")
            else:
                print("  No retriever available (no zones loaded)")
    else:
        print("\nUsage:")
        print("  python location_rag.py --list              # List all zones")
        print("  python location_rag.py --test LON LAT      # Test coordinates")
        print("  python location_rag.py --search QUERY      # Semantic search")
        print("  python location_rag.py --retriever QUERY   # LangChain retriever demo")
        print("\nExample:")
        print("  python location_rag.py --test -122.0005 37.0015")
        print("  python location_rag.py --search 'safe place for a cat'")