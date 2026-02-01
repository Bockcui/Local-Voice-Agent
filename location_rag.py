"""
Location RAG System with Vector Search and Polygon Geofencing
==============================================================
Uses ChromaDB for semantic search and Shapely for polygon geometry.
Provides rich, contextual location descriptions for Ranni's GPS coordinates.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

import chromadb
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import nearest_points
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LocationZone:
    """
    A geographic zone defined by a convex polygon with rich metadata.
    """
    id: str
    name: str
    description: str  # Rich description for semantic search
    polygon_coords: List[Tuple[float, float]]  # List of (lon, lat) tuples forming the polygon
    zone_type: str = "general"  # home, yard, neighbor, street, park, danger, etc.
    safety_level: str = "safe"  # safe, caution, danger
    typical_activities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    time_context: Dict[str, str] = field(default_factory=dict)  # e.g., {"night": "dangerous after dark"}
    
    def to_shapely(self) -> Polygon:
        """Convert to Shapely Polygon."""
        return Polygon(self.polygon_coords)
    
    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if a point is inside this zone."""
        point = Point(lon, lat)
        return self.to_shapely().contains(point)
    
    def distance_to_point(self, lon: float, lat: float) -> float:
        """Get distance from point to nearest edge of polygon (in degrees)."""
        point = Point(lon, lat)
        polygon = self.to_shapely()
        if polygon.contains(point):
            return 0.0
        return point.distance(polygon)


@dataclass 
class LocationContext:
    """
    Rich context about a location query result.
    """
    zone: Optional[LocationZone]
    is_inside_zone: bool
    distance_m: float
    semantic_matches: List[Dict[str, Any]]
    narrative: str
    safety_alert: Optional[str] = None
    coordinates: Tuple[float, float] = (0.0, 0.0)
    timestamp: str = ""


# ============================================================================
# LOCATION RAG DATABASE
# ============================================================================

class LocationRAG:
    """
    RAG system for location-aware responses using vector search and polygon geofencing.
    """
    
    # Approximate meters per degree at mid-latitudes
    METERS_PER_DEGREE = 111000
    
    def __init__(self, db_path: str = "./ranni_location_db"):
        """
        Initialize the Location RAG system.
        
        Args:
            db_path: Path to persist ChromaDB data
        """
        self.db_path = db_path
        self.zones: Dict[str, LocationZone] = {}
        
        # Initialize ChromaDB with the new API (v0.4+)
        # Use PersistentClient for data that persists to disk
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Create or get the collection for location embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="location_zones",
            metadata={"description": "Semantic embeddings for location zones"}
        )
        
        # Load existing zones from JSON if available
        self._load_zones_from_disk()
        
        logger.info(f"LocationRAG initialized with DB at {db_path}")
    
    def add_zone(self, zone: LocationZone) -> None:
        """
        Add a location zone to the RAG system.
        
        Args:
            zone: LocationZone to add
        """
        # Store in local dict for polygon operations
        self.zones[zone.id] = zone
        
        # Create rich text for embedding
        embedding_text = self._create_embedding_text(zone)
        
        # Create metadata for filtering
        metadata = {
            "name": zone.name,
            "zone_type": zone.zone_type,
            "safety_level": zone.safety_level,
            "tags": ",".join(zone.tags),
            "polygon_json": json.dumps(zone.polygon_coords)
        }
        
        # Add or update in ChromaDB
        self.collection.upsert(
            ids=[zone.id],
            documents=[embedding_text],
            metadatas=[metadata]
        )
        
        logger.info(f"Added zone: {zone.name} ({zone.id})")
    
    def _create_embedding_text(self, zone: LocationZone) -> str:
        """Create rich text for semantic embedding."""
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
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone from the system."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.collection.delete(ids=[zone_id])
            return True
        return False
    
    def find_containing_zone(self, lon: float, lat: float) -> Optional[LocationZone]:
        """
        Find which zone contains the given point.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            The containing zone or None
        """
        point = Point(lon, lat)
        
        for zone in self.zones.values():
            if zone.to_shapely().contains(point):
                return zone
        
        return None
    
    def find_nearest_zone(self, lon: float, lat: float) -> Tuple[Optional[LocationZone], float]:
        """
        Find the nearest zone to a point.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Tuple of (nearest zone, distance in meters)
        """
        point = Point(lon, lat)
        nearest_zone = None
        min_distance = float('inf')
        
        for zone in self.zones.values():
            polygon = zone.to_shapely()
            
            if polygon.contains(point):
                return zone, 0.0
            
            distance = point.distance(polygon)
            if distance < min_distance:
                min_distance = distance
                nearest_zone = zone
        
        # Convert degrees to approximate meters
        distance_m = min_distance * self.METERS_PER_DEGREE
        
        return nearest_zone, distance_m
    
    def semantic_search(self, query: str, n_results: int = 3, 
                       filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on location descriptions.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            filter_type: Optional zone_type filter
            
        Returns:
            List of matching zones with scores
        """
        where_filter = None
        if filter_type:
            where_filter = {"zone_type": filter_type}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        matches = []
        if results and results['ids'] and results['ids'][0]:
            for i, zone_id in enumerate(results['ids'][0]):
                if zone_id in self.zones:
                    matches.append({
                        "zone": self.zones[zone_id],
                        "score": results['distances'][0][i] if results['distances'] else 0,
                        "document": results['documents'][0][i] if results['documents'] else ""
                    })
        
        return matches
    
    def get_location_context(self, lon: float, lat: float, 
                            semantic_query: Optional[str] = None) -> LocationContext:
        """
        Get rich context about a location combining spatial and semantic search.
        
        Args:
            lon: Longitude
            lat: Latitude
            semantic_query: Optional query for semantic enhancement
            
        Returns:
            LocationContext with full details
        """
        timestamp = datetime.now().isoformat()
        current_hour = datetime.now().hour
        
        # Check if inside any zone
        containing_zone = self.find_containing_zone(lon, lat)
        
        if containing_zone:
            zone = containing_zone
            is_inside = True
            distance_m = 0.0
        else:
            zone, distance_m = self.find_nearest_zone(lon, lat)
            is_inside = False
        
        # Perform semantic search if query provided or use default
        query = semantic_query or "cat location whereabouts position"
        semantic_matches = self.semantic_search(query, n_results=3)
        
        # Generate narrative
        narrative = self._generate_narrative(
            zone=zone,
            is_inside=is_inside,
            distance_m=distance_m,
            hour=current_hour,
            lon=lon,
            lat=lat
        )
        
        # Check for safety alerts
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
            timestamp=timestamp
        )
    
    def _generate_narrative(self, zone: Optional[LocationZone], is_inside: bool,
                           distance_m: float, hour: int, lon: float, lat: float) -> str:
        """Generate a dramatic narrative description."""
        
        if zone is None:
            return (
                f"The feline wanders in uncharted territory, beyond the boundaries of mapped lands. "
                f"Coordinates: {lat:.6f}, {lon:.6f}"
            )
        
        # Determine time context
        if 6 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 17:
            time_period = "afternoon"
        elif 17 <= hour < 21:
            time_period = "evening"
        else:
            time_period = "night"
        
        # Get time-specific context if available
        time_context = zone.time_context.get(time_period, "")
        
        if is_inside:
            narrative = f"Ranni dwells within {zone.name} - {zone.description}."
            
            if zone.typical_activities:
                activity = zone.typical_activities[0]
                narrative += f" She is likely {activity}."
            
            if time_context:
                narrative += f" {time_context}"
                
        else:
            # Calculate direction
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
                f"Ranni prowls {proximity} {zone.name}, approximately {int(distance_m)} meters "
                f"to the {direction}. {zone.description}."
            )
        
        # Add safety context
        if zone.safety_level == "danger":
            narrative += " DANGER: This area poses risks to her safety!"
        elif zone.safety_level == "caution":
            narrative += " Caution is advised in this area."
        
        return narrative
    
    def _get_direction_to_zone(self, lon: float, lat: float, zone: LocationZone) -> str:
        """Get cardinal direction from point to zone centroid."""
        polygon = zone.to_shapely()
        centroid = polygon.centroid
        
        delta_lon = centroid.x - lon
        delta_lat = centroid.y - lat
        
        if abs(delta_lat) > abs(delta_lon):
            return "north" if delta_lat > 0 else "south"
        else:
            return "east" if delta_lon > 0 else "west"
    
    def persist(self) -> None:
        """Persist the zone data to disk (ChromaDB auto-persists, this saves zone JSON)."""
        # ChromaDB PersistentClient auto-persists, so we just save zones to JSON
        zones_data = {zone_id: asdict(zone) for zone_id, zone in self.zones.items()}
        zones_path = os.path.join(self.db_path, "zones.json")
        
        os.makedirs(self.db_path, exist_ok=True)
        with open(zones_path, 'w') as f:
            json.dump(zones_data, f, indent=2)
        
        logger.info("Zone data persisted")
    
    def _load_zones_from_disk(self) -> None:
        """Load zones from the JSON file if it exists."""
        zones_path = os.path.join(self.db_path, "zones.json")
        
        if os.path.exists(zones_path):
            try:
                with open(zones_path, 'r') as f:
                    zones_data = json.load(f)
                
                for zone_id, data in zones_data.items():
                    zone = LocationZone(
                        id=data.get("id", zone_id),
                        name=data["name"],
                        description=data["description"],
                        polygon_coords=[tuple(coord) for coord in data["polygon_coords"]],
                        zone_type=data.get("zone_type", "general"),
                        safety_level=data.get("safety_level", "safe"),
                        typical_activities=data.get("typical_activities", []),
                        tags=data.get("tags", []),
                        time_context=data.get("time_context", {})
                    )
                    # Add to local dict only (ChromaDB already has it)
                    self.zones[zone.id] = zone
                
                logger.info(f"Loaded {len(self.zones)} zones from disk")
            except Exception as e:
                logger.warning(f"Could not load zones from disk: {e}")
    
    def load_zones_from_json(self, json_path: str) -> int:
        """
        Load zones from a JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Number of zones loaded
        """
        with open(json_path, 'r') as f:
            zones_data = json.load(f)
        
        count = 0
        for zone_id, data in zones_data.items():
            zone = LocationZone(
                id=data.get("id", zone_id),
                name=data["name"],
                description=data["description"],
                polygon_coords=[tuple(coord) for coord in data["polygon_coords"]],
                zone_type=data.get("zone_type", "general"),
                safety_level=data.get("safety_level", "safe"),
                typical_activities=data.get("typical_activities", []),
                tags=data.get("tags", []),
                time_context=data.get("time_context", {})
            )
            self.add_zone(zone)
            count += 1
        
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
    
    Note: Coordinates are (longitude, latitude) for Shapely compatibility!
    """
    
    zones = [
        # === EXAMPLE ===
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

# Global instance - initialize once
_location_rag: Optional[LocationRAG] = None


def get_location_rag() -> LocationRAG:
    """Get or create the global LocationRAG instance."""
    global _location_rag
    
    if _location_rag is None:
        _location_rag = LocationRAG(db_path="./ranni_location_db")
        
        # Check if zones exist, if not, load examples
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
    
    print("Location RAG System")
    print("=" * 50)
    
    # Initialize
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
            # Test with coordinates: python location_rag.py --test -122.0005 37.0015
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
    else:
        print("\nUsage:")
        print("  python location_rag.py --list              # List all zones")
        print("  python location_rag.py --test LON LAT      # Test coordinates")
        print("  python location_rag.py --search QUERY      # Semantic search")
        print("\nExample:")
        print("  python location_rag.py --test -122.0005 37.0015")
        print("  python location_rag.py --search 'safe place for a cat'")