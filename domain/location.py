"""
Location model representing delivery points (customers) and depot.
Implements geographical coordinates, demand characteristics, and service constraints.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Location:
    """
    Represents a physical location for delivery operations.
    
    Attributes:
        id: Unique identifier for the location
        x: X-coordinate (longitude or grid position)
        y: Y-coordinate (latitude or grid position)
        name: Human-readable name (optional)
        demand_weight: Weight of goods to deliver/pickup (kg)
        service_time: Time required to service this location (minutes)
        is_depot: Whether this location is the starting/ending depot
        time_window_start: Start of delivery time window (minutes from midnight)
        time_window_end: End of delivery time window (minutes from midnight)
    """
    
    id: int
    x: float
    y: float
    demand_weight: float = 0.0
    service_time: float = 5.0  # Default 5 minutes service time
    name: Optional[str] = None
    is_depot: bool = False
    time_window_start: float = 0.0
    time_window_end: float = 1440.0  # End of day (24h * 60)
    
    def __post_init__(self):
        """Validate location parameters after initialization."""
        if self.demand_weight < 0:
            raise ValueError(f"Demand weight cannot be negative for location {self.id}")
        if self.service_time < 0:
            raise ValueError(f"Service time cannot be negative for location {self.id}")
        if self.time_window_end <= self.time_window_start:
            raise ValueError(f"Time window end must be after start for location {self.id}")
    
    def distance_to(self, other: 'Location') -> float:
        """
        Calculate Euclidean distance to another location.
        
        Args:
            other: Another Location instance
            
        Returns:
            Euclidean distance between the two locations
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def time_to(self, other: 'Location', speed_kmh: float = 30.0) -> float:
        """
        Calculate travel time to another location.
        
        Args:
            other: Destination location
            speed_kmh: Average travel speed in km/h
            
        Returns:
            Travel time in minutes
        """
        distance_km = self.distance_to(other)  # Assuming coordinates in km
        speed_kmpm = speed_kmh / 60.0  # Convert to km per minute
        return distance_km / speed_kmpm if speed_kmpm > 0 else 0.0
    
    def is_within_time_window(self, arrival_time: float) -> bool:
        """
        Check if arrival time is within the location's time window.
        
        Args:
            arrival_time: Proposed arrival time (minutes from midnight)
            
        Returns:
            True if arrival time is within acceptable window
        """
        return self.time_window_start <= arrival_time <= self.time_window_end
    
    def __hash__(self):
        """Make Location hashable for use in sets and dictionaries."""
        return hash(self.id)
    
    def __eq__(self, other):
        """Compare locations based on ID."""
        if not isinstance(other, Location):
            return False
        return self.id == other.id