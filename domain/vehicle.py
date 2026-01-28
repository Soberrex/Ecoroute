"""
Vehicle model representing delivery vehicles in the fleet.
Handles capacity constraints, fuel efficiency, and operational limits.
"""

from dataclasses import dataclass
from typing import List, Optional
import uuid

from domain.location import Location


@dataclass
class Vehicle:
    """
    Represents a delivery vehicle with operational constraints.
    
    Attributes:
        id: Unique vehicle identifier
        max_capacity: Maximum weight capacity (kg)
        fuel_efficiency: Fuel consumption rate (liters per km)
        max_route_time: Maximum allowable route duration (minutes)
        fixed_cost: Fixed cost per deployment ($)
        variable_cost: Cost per km traveled ($/km)
        current_load: Current weight load (kg)
        current_location: Current position of the vehicle
        fuel_cost_per_liter: Cost of fuel ($/liter)
    """
    
    max_capacity: float
    fuel_efficiency: float  # liters per km
    max_route_time: float  # minutes
    id: str = None
    fixed_cost: float = 50.0  # Default fixed cost per deployment
    variable_cost: float = 0.5  # Default $0.5 per km
    current_load: float = 0.0
    current_location: Optional[Location] = None
    fuel_cost_per_liter: float = 1.5  # Default fuel price
    
    def __post_init__(self):
        """Initialize vehicle with unique ID if not provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())[:8]  # Short unique ID
        
        # Validate vehicle parameters
        if self.max_capacity <= 0:
            raise ValueError(f"Vehicle {self.id}: max_capacity must be positive")
        if self.fuel_efficiency <= 0:
            raise ValueError(f"Vehicle {self.id}: fuel_efficiency must be positive")
        if self.max_route_time <= 0:
            raise ValueError(f"Vehicle {self.id}: max_route_time must be positive")
        if self.fixed_cost < 0:
            raise ValueError(f"Vehicle {self.id}: fixed_cost cannot be negative")
        if self.variable_cost < 0:
            raise ValueError(f"Vehicle {self.id}: variable_cost cannot be negative")
    
    def can_load(self, weight: float) -> bool:
        """
        Check if the vehicle can load additional weight.
        
        Args:
            weight: Weight to add (kg)
            
        Returns:
            True if weight can be added without exceeding capacity
        """
        return self.current_load + weight <= self.max_capacity
    
    def load(self, weight: float) -> None:
        """
        Add weight to the vehicle's current load.
        
        Args:
            weight: Weight to add (kg)
            
        Raises:
            ValueError: If loading would exceed capacity
        """
        if not self.can_load(weight):
            raise ValueError(
                f"Cannot load {weight}kg. "
                f"Current load: {self.current_load}kg, "
                f"Capacity: {self.max_capacity}kg"
            )
        self.current_load += weight
    
    def unload(self, weight: float) -> None:
        """
        Remove weight from the vehicle's current load.
        
        Args:
            weight: Weight to remove (kg)
            
        Raises:
            ValueError: If unloading would result in negative load
        """
        if self.current_load - weight < 0:
            raise ValueError(
                f"Cannot unload {weight}kg. Current load: {self.current_load}kg"
            )
        self.current_load -= weight
    
    def reset_load(self) -> None:
        """Reset the vehicle's load to zero."""
        self.current_load = 0.0
    
    def calculate_fuel_consumption(self, distance_km: float) -> float:
        """
        Calculate fuel consumption for a given distance.
        
        Args:
            distance_km: Distance to travel in kilometers
            
        Returns:
            Fuel consumption in liters
        """
        return distance_km * self.fuel_efficiency
    
    def calculate_fuel_cost(self, distance_km: float) -> float:
        """
        Calculate fuel cost for a given distance.
        
        Args:
            distance_km: Distance to travel in kilometers
            
        Returns:
            Fuel cost in dollars
        """
        fuel_consumption = self.calculate_fuel_consumption(distance_km)
        return fuel_consumption * self.fuel_cost_per_liter
    
    def calculate_total_cost(self, distance_km: float) -> float:
        """
        Calculate total operational cost for a given distance.
        
        Args:
            distance_km: Distance to travel in kilometers
            
        Returns:
            Total cost in dollars (fixed + variable + fuel)
        """
        distance_cost = distance_km * self.variable_cost
        fuel_cost = self.calculate_fuel_cost(distance_km)
        return self.fixed_cost + distance_cost + fuel_cost
    
    def is_route_feasible(self, total_distance: float, total_time: float) -> bool:
        """
        Check if a route is feasible for this vehicle.
        
        Args:
            total_distance: Total route distance (km)
            total_time: Total route time (minutes)
            
        Returns:
            True if route is within vehicle constraints
        """
        return total_time <= self.max_route_time
    
    def __str__(self):
        """String representation of the vehicle."""
        return (f"Vehicle {self.id}: "
                f"Capacity={self.max_capacity}kg, "
                f"Fuel={self.fuel_efficiency}L/km, "
                f"MaxTime={self.max_route_time}min")