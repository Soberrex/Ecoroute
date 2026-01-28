"""
Traffic simulation for dynamic route cost calculations.
Models traffic zones, peak hours, and dynamic conditions.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from domain.location import Location
from domain.route import Route
from cost.cost_engine import CostEngine, CostComponent, Constraint


class TrafficZone:
    """
    Represents a traffic zone with increased travel times.
    """
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 slowdown_factor: float = 1.5, name: str = ""):
        """
        Initialize traffic zone.
        
        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate
            y_min: Minimum y coordinate
            y_max: Maximum y coordinate
            slowdown_factor: Travel time multiplier in this zone
            name: Zone name for identification
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.slowdown_factor = slowdown_factor
        self.name = name
    
    def contains(self, x: float, y: float) -> bool:
        """
        Check if point is inside traffic zone.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside zone
        """
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def segment_intersection(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[bool, float]:
        """
        Check if line segment intersects with zone.
        
        Args:
            x1, y1: Start point coordinates
            x2, y2: End point coordinates
            
        Returns:
            Tuple of (intersects, fraction_of_segment_in_zone)
        """
        # Simplified implementation: check if either endpoint is in zone
        # For production, implement proper line-rectangle intersection
        
        start_in = self.contains(x1, y1)
        end_in = self.contains(x2, y2)
        
        if start_in and end_in:
            return True, 1.0
        elif start_in or end_in:
            return True, 0.5  # Approximate
        else:
            # Check if segment crosses zone (simplified)
            # This is a placeholder - implement proper intersection for production
            return False, 0.0


class PeakHour:
    """
    Represents peak hour periods with increased traffic.
    """
    
    def __init__(self, start_hour: int, start_minute: int, 
                 end_hour: int, end_minute: int, slowdown_factor: float = 1.8):
        """
        Initialize peak hour period.
        
        Args:
            start_hour: Start hour (0-23)
            start_minute: Start minute (0-59)
            end_hour: End hour (0-23)
            end_minute: End minute (0-59)
            slowdown_factor: Travel time multiplier during peak hours
        """
        self.start_time = start_hour * 60 + start_minute  # Convert to minutes from midnight
        self.end_time = end_hour * 60 + end_minute
        self.slowdown_factor = slowdown_factor
    
    def is_peak_time(self, current_time: float) -> bool:
        """
        Check if current time is within peak hours.
        
        Args:
            current_time: Time in minutes from midnight
            
        Returns:
            True if within peak hours
        """
        return self.start_time <= current_time <= self.end_time


class TrafficSimulator:
    """
    Simulates traffic conditions for route evaluation.
    """
    
    def __init__(self, traffic_zones: List[TrafficZone] = None,
                 peak_hours: List[PeakHour] = None):
        """
        Initialize traffic simulator.
        
        Args:
            traffic_zones: List of traffic zones
            peak_hours: List of peak hour periods
        """
        self.traffic_zones = traffic_zones or []
        self.peak_hours = peak_hours or []
        
        # Default peak hours: morning 7-9, evening 5-7
        if not self.peak_hours:
            self.peak_hours = [
                PeakHour(7, 0, 9, 0, 1.8),   # Morning peak
                PeakHour(17, 0, 19, 0, 1.5)  # Evening peak
            ]
    
    def calculate_traffic_factor(self, x1: float, y1: float, x2: float, y2: float,
                                current_time: float = 480.0) -> float:
        """
        Calculate traffic slowdown factor for a route segment.
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            current_time: Current time in minutes from midnight
            
        Returns:
            Traffic slowdown factor (1.0 = normal, >1.0 = slower)
        """
        base_factor = 1.0
        
        # Apply peak hour factor
        for peak_hour in self.peak_hours:
            if peak_hour.is_peak_time(current_time):
                base_factor = max(base_factor, peak_hour.slowdown_factor)
        
        # Apply traffic zone factors
        for zone in self.traffic_zones:
            intersects, fraction = zone.segment_intersection(x1, y1, x2, y2)
            if intersects:
                # Weighted average based on fraction in zone
                zone_factor = 1.0 + (zone.slowdown_factor - 1.0) * fraction
                base_factor = max(base_factor, zone_factor)
        
        return base_factor
    
    def simulate_route(self, route: Route, start_time: float = 480.0) -> Dict[str, Any]:
        """
        Simulate route with traffic conditions.
        
        Args:
            route: Route to simulate
            start_time: Departure time from depot
            
        Returns:
            Dictionary with simulation results
        """
        if not route.locations:
            return {
                'total_time': 0.0,
                'traffic_delay': 0.0,
                'adjusted_time': 0.0,
                'traffic_factors': []
            }
        
        current_time = start_time
        current_location = route.depot
        traffic_factors = []
        total_traffic_delay = 0.0
        
        for location in route.locations:
            # Calculate base travel time
            base_travel_time = current_location.time_to(location, route.speed_kmh)
            
            # Calculate traffic factor
            traffic_factor = self.calculate_traffic_factor(
                current_location.x, current_location.y,
                location.x, location.y,
                current_time
            )
            traffic_factors.append(traffic_factor)
            
            # Calculate adjusted travel time
            adjusted_travel_time = base_travel_time * traffic_factor
            traffic_delay = adjusted_travel_time - base_travel_time
            total_traffic_delay += traffic_delay
            
            # Update current time and location
            current_time += adjusted_travel_time
            current_time += location.service_time
            current_location = location
        
        # Return trip to depot
        base_return_time = current_location.time_to(route.depot, route.speed_kmh)
        traffic_factor = self.calculate_traffic_factor(
            current_location.x, current_location.y,
            route.depot.x, route.depot.y,
            current_time
        )
        
        adjusted_return_time = base_return_time * traffic_factor
        total_traffic_delay += adjusted_return_time - base_return_time
        
        # Total route time with traffic
        total_adjusted_time = (current_time + adjusted_return_time) - start_time
        
        return {
            'total_time': total_adjusted_time,
            'traffic_delay': total_traffic_delay,
            'adjusted_time': total_adjusted_time,
            'traffic_factors': traffic_factors,
            'peak_hour_penalty': self._calculate_peak_hour_penalty(start_time, total_adjusted_time)
        }
    
    def _calculate_peak_hour_penalty(self, start_time: float, total_time: float) -> float:
        """
        Calculate penalty for traveling during peak hours.
        
        Args:
            start_time: Route start time
            total_time: Total route time
            
        Returns:
            Peak hour penalty factor
        """
        penalty = 0.0
        current_time = start_time
        
        # Sample time points along route
        sample_points = 10
        for i in range(sample_points + 1):
            sample_time = start_time + (total_time * i / sample_points)
            
            for peak_hour in self.peak_hours:
                if peak_hour.is_peak_time(sample_time):
                    penalty += (peak_hour.slowdown_factor - 1.0) / (sample_points + 1)
        
        return penalty


class DynamicTraffic(CostComponent):
    """
    Dynamic traffic cost component for CostEngine.
    """
    
    def __init__(self, traffic_simulator: TrafficSimulator, 
                 cost_per_minute_delay: float = 0.5, weight: float = 1.0):
        """
        Initialize dynamic traffic cost component.
        
        Args:
            traffic_simulator: Traffic simulator instance
            cost_per_minute_delay: Cost per minute of traffic delay
            weight: Component weight
        """
        super().__init__("DynamicTrafficCost", weight)
        self.traffic_simulator = traffic_simulator
        self.cost_per_minute_delay = cost_per_minute_delay
    
    def calculate(self, route: Route, **kwargs) -> float:
        """Calculate traffic-related costs."""
        start_time = kwargs.get('start_time', 480.0)
        
        # Simulate route with traffic
        simulation = self.traffic_simulator.simulate_route(route, start_time)
        
        # Calculate cost based on traffic delay
        traffic_delay = simulation.get('traffic_delay', 0.0)
        traffic_cost = traffic_delay * self.cost_per_minute_delay
        
        # Additional penalty for peak hour travel
        peak_penalty = simulation.get('peak_hour_penalty', 0.0) * 2.0
        
        return (traffic_cost + peak_penalty) * self.weight


class TrafficAwareConstraint(Constraint):
    """
    Constraint that considers traffic conditions.
    """
    
    def __init__(self, traffic_simulator: TrafficSimulator, 
                 max_traffic_delay: float = 60.0, penalty_weight: float = 5.0):
        """
        Initialize traffic-aware constraint.
        
        Args:
            traffic_simulator: Traffic simulator instance
            max_traffic_delay: Maximum acceptable traffic delay (minutes)
            penalty_weight: Penalty weight
        """
        super().__init__("TrafficDelayConstraint", penalty_weight)
        self.traffic_simulator = traffic_simulator
        self.max_traffic_delay = max_traffic_delay
    
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """Check traffic delay constraint."""
        start_time = kwargs.get('start_time', 480.0)
        
        simulation = self.traffic_simulator.simulate_route(route, start_time)
        traffic_delay = simulation.get('traffic_delay', 0.0)
        
        if traffic_delay <= self.max_traffic_delay:
            return True, 0.0
        else:
            violation = traffic_delay - self.max_traffic_delay
            penalty = self.penalty_weight * (violation ** 2)
            return False, penalty