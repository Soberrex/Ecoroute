"""
Route model representing a delivery route for a single vehicle.
Handles route calculations, feasibility checks, and load management.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from domain.location import Location
from domain.vehicle import Vehicle
import math


@dataclass
class Route:
    """
    Represents a delivery route for a single vehicle.
    
    Attributes:
        vehicle: Vehicle assigned to this route
        locations: Ordered list of locations to visit (excluding depot)
        depot: Starting and ending depot location
        distance_cache: Cached total distance (for performance)
        time_cache: Cached total time (for performance)
        speed_kmh: Average travel speed (km/h)
    """
    
    vehicle: Vehicle
    depot: Location
    locations: List[Location] = field(default_factory=list)
    speed_kmh: float = 30.0  # Default average speed 30 km/h
    
    # Performance caches (lazy evaluation)
    _total_distance: Optional[float] = field(default=None, init=False, repr=False)
    _total_time: Optional[float] = field(default=None, init=False, repr=False)
    _arrival_times: Optional[List[float]] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate route configuration."""
        if not self.depot.is_depot:
            raise ValueError("Route must start and end at a depot")
        
        # Set vehicle's starting location to depot
        self.vehicle.current_location = self.depot
        
        # Reset distance and time caches if locations change
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate cached calculations."""
        self._total_distance = None
        self._total_time = None
        self._arrival_times = None
    
    def add_location(self, location: Location, position: Optional[int] = None) -> None:
        """
        Add a location to the route.
        
        Args:
            location: Location to add
            position: Position in the route (None for end)
            
        Raises:
            ValueError: If location is the depot
        """
        if location.is_depot:
            raise ValueError("Cannot add depot to route (already implicitly included)")
        
        if position is None:
            self.locations.append(location)
        else:
            self.locations.insert(position, location)
        
        self._invalidate_cache()
    
    def remove_location(self, position: int) -> Location:
        """
        Remove a location from the route.
        
        Args:
            position: Position of location to remove
            
        Returns:
            The removed location
            
        Raises:
            IndexError: If position is invalid
        """
        if position < 0 or position >= len(self.locations):
            raise IndexError(f"Invalid position {position} in route")
        
        location = self.locations.pop(position)
        self._invalidate_cache()
        return location
    
    def swap_locations(self, pos1: int, pos2: int) -> None:
        """
        Swap two locations in the route.
        
        Args:
            pos1: Position of first location
            pos2: Position of second location
        """
        self.locations[pos1], self.locations[pos2] = self.locations[pos2], self.locations[pos1]
        self._invalidate_cache()
    
    def calculate_total_distance(self, force_recalculate: bool = False) -> float:
        """
        Calculate total route distance (starting and ending at depot).
        
        Args:
            force_recalculate: Force recalculation even if cached
            
        Returns:
            Total distance in kilometers
        """
        if self._total_distance is not None and not force_recalculate:
            return self._total_distance
        
        if not self.locations:
            self._total_distance = 0.0
            return 0.0
        
        # Start from depot
        total_distance = 0.0
        current = self.depot
        
        # Travel to each location
        for location in self.locations:
            total_distance += current.distance_to(location)
            current = location
        
        # Return to depot
        total_distance += current.distance_to(self.depot)
        
        self._total_distance = total_distance
        return total_distance
    
    def calculate_arrival_times(self, start_time: float = 480.0) -> List[float]:
        """
        Calculate arrival times at each location.
        
        Args:
            start_time: Departure time from depot (minutes from midnight)
            
        Returns:
            List of arrival times for each location
        """
        if self._arrival_times is not None:
            return self._arrival_times
        
        if not self.locations:
            self._arrival_times = []
            return []
        
        arrival_times = []
        current_time = start_time
        current_location = self.depot
        
        for location in self.locations:
            # Travel time to next location
            travel_time = current_location.time_to(location, self.speed_kmh)
            current_time += travel_time
            
            # Service time at location
            current_time += location.service_time
            
            arrival_times.append(current_time)
            current_location = location
        
        self._arrival_times = arrival_times
        return arrival_times
    
    def calculate_total_time(self, start_time: float = 480.0, 
                           include_service_time: bool = True) -> float:
        """
        Calculate total route time.
        
        Args:
            start_time: Departure time from depot
            include_service_time: Whether to include service time at locations
            
        Returns:
            Total route time in minutes
        """
        if self._total_time is not None:
            return self._total_time
        
        if not self.locations:
            self._total_time = 0.0
            return 0.0
        
        # Get arrival times
        arrival_times = self.calculate_arrival_times(start_time)
        
        if not arrival_times:
            self._total_time = 0.0
            return 0.0
        
        # Time from last location back to depot
        last_location = self.locations[-1]
        return_travel_time = last_location.time_to(self.depot, self.speed_kmh)
        
        # Total time is from start to return to depot
        last_arrival = arrival_times[-1]
        total_time = last_arrival + return_travel_time - start_time
        
        if not include_service_time:
            # Subtract service times
            total_service_time = sum(loc.service_time for loc in self.locations)
            total_time -= total_service_time
        
        self._total_time = total_time
        return total_time
    
    def calculate_total_load(self) -> float:
        """
        Calculate total weight demand on the route.
        
        Returns:
            Total weight in kilograms
        """
        return sum(location.demand_weight for location in self.locations)
    
    def check_capacity_constraint(self) -> Tuple[bool, float, float]:
        """
        Check if route satisfies vehicle capacity constraint.
        
        Returns:
            Tuple of (is_feasible, total_load, capacity_violation)
        """
        total_load = self.calculate_total_load()
        capacity_violation = max(0, total_load - self.vehicle.max_capacity)
        is_feasible = capacity_violation <= 1e-6  # Small tolerance
        
        return is_feasible, total_load, capacity_violation
    
    def check_time_window_constraint(self, start_time: float = 480.0) -> Tuple[bool, int, float]:
        """
        Check if route satisfies all time window constraints.
        
        Args:
            start_time: Departure time from depot
            
        Returns:
            Tuple of (is_feasible, violations_count, total_penalty_minutes)
        """
        arrival_times = self.calculate_arrival_times(start_time)
        violations = 0
        total_penalty = 0.0
        
        for location, arrival in zip(self.locations, arrival_times):
            if not location.is_within_time_window(arrival):
                violations += 1
                # Calculate how far outside the window
                if arrival < location.time_window_start:
                    penalty = location.time_window_start - arrival
                else:
                    penalty = arrival - location.time_window_end
                total_penalty += penalty
        
        is_feasible = violations == 0
        return is_feasible, violations, total_penalty
    
    def check_route_time_constraint(self, start_time: float = 480.0) -> Tuple[bool, float, float]:
        """
        Check if route satisfies maximum route time constraint.
        
        Args:
            start_time: Departure time from depot
            
        Returns:
            Tuple of (is_feasible, total_time, time_violation)
        """
        total_time = self.calculate_total_time(start_time)
        time_violation = max(0, total_time - self.vehicle.max_route_time)
        is_feasible = time_violation <= 1e-6  # Small tolerance
        
        return is_feasible, total_time, time_violation
    
    def is_feasible(self, start_time: float = 480.0) -> Tuple[bool, List[str]]:
        """
        Check if route is feasible considering all constraints.
        
        Args:
            start_time: Departure time from depot
            
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []
        
        # Check capacity
        capacity_feasible, load, capacity_violation = self.check_capacity_constraint()
        if not capacity_feasible:
            violations.append(
                f"Capacity exceeded: {load:.1f}kg > {self.vehicle.max_capacity:.1f}kg "
                f"(violation: {capacity_violation:.1f}kg)"
            )
        
        # Check time windows
        time_window_feasible, window_violations, window_penalty = \
            self.check_time_window_constraint(start_time)
        if not time_window_feasible:
            violations.append(
                f"Time window violations: {window_violations} locations "
                f"(total penalty: {window_penalty:.1f}min)"
            )
        
        # Check route time
        time_feasible, total_time, time_violation = \
            self.check_route_time_constraint(start_time)
        if not time_feasible:
            violations.append(
                f"Route time exceeded: {total_time:.1f}min > "
                f"{self.vehicle.max_route_time:.1f}min "
                f"(violation: {time_violation:.1f}min)"
            )
        
        is_feasible = capacity_feasible and time_window_feasible and time_feasible
        return is_feasible, violations
    
    def get_route_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get coordinates of the complete route (including depot at start and end).
        
        Returns:
            List of (x, y) coordinates
        """
        coords = [(self.depot.x, self.depot.y)]
        coords.extend([(loc.x, loc.y) for loc in self.locations])
        coords.append((self.depot.x, self.depot.y))  # Return to depot
        return coords
    
    def __len__(self) -> int:
        """Number of delivery locations in the route."""
        return len(self.locations)
    
    def __str__(self) -> str:
        """String representation of the route."""
        if not self.locations:
            return f"Empty route for {self.vehicle}"
        
        locations_str = " -> ".join(str(loc.id) for loc in self.locations)
        distance = self.calculate_total_distance()
        load = self.calculate_total_load()
        
        return (f"Route for {self.vehicle}: "
                f"Depot{self.depot.id} -> {locations_str} -> Depot{self.depot.id} | "
                f"Distance: {distance:.1f}km, Load: {load:.1f}kg")