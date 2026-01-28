"""
Cost Engine for evaluating route costs with multiple components.
Implements the Open/Closed Principle for easy extension of new cost components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import math
from domain.route import Route
from domain.location import Location
from domain.vehicle import Vehicle


@dataclass
class CostComponent(ABC):
    """
    Abstract base class for cost components.
    Each component calculates a specific type of cost or penalty.
    """
    
    name: str
    weight: float = 1.0
    
    @abstractmethod
    def calculate(self, route: Route, **kwargs) -> float:
        """
        Calculate the cost component value.
        
        Args:
            route: Route to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Cost component value
        """
        pass
    
    def __str__(self):
        return f"{self.name} (weight: {self.weight})"


@dataclass
class Constraint(ABC):
    """
    Abstract base class for route constraints.
    """
    
    name: str
    penalty_weight: float = 100.0
    
    @abstractmethod
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """
        Check constraint and return penalty if violated.
        
        Args:
            route: Route to check
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (is_satisfied, penalty_value)
        """
        pass


class DistanceCostComponent(CostComponent):
    """
    Calculates distance-based costs including fuel consumption.
    """
    
    def __init__(self, distance_cost_per_km: float = 0.5, weight: float = 1.0):
        super().__init__("DistanceCost", weight)
        self.distance_cost_per_km = distance_cost_per_km
    
    def calculate(self, route: Route, **kwargs) -> float:
        """Calculate distance and fuel costs."""
        distance_km = route.calculate_total_distance()
        
        # Base distance cost
        distance_cost = distance_km * self.distance_cost_per_km
        
        # Fuel cost from vehicle
        vehicle = route.vehicle
        fuel_cost = vehicle.calculate_fuel_cost(distance_km)
        
        return (distance_cost + fuel_cost) * self.weight


class TimeCostComponent(CostComponent):
    """
    Calculates time-based costs including driver wages and vehicle usage.
    """
    
    def __init__(self, cost_per_minute: float = 0.25, weight: float = 1.0):
        super().__init__("TimeCost", weight)
        self.cost_per_minute = cost_per_minute
    
    def calculate(self, route: Route, **kwargs) -> float:
        """Calculate time-based costs."""
        total_time = route.calculate_total_time()
        return total_time * self.cost_per_minute * self.weight


class CapacityConstraint(Constraint):
    """
    Penalizes routes that exceed vehicle capacity.
    """
    
    def __init__(self, penalty_weight: float = 100.0):
        super().__init__("CapacityConstraint", penalty_weight)
    
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """Check capacity constraint."""
        is_feasible, total_load, violation = route.check_capacity_constraint()
        
        if is_feasible:
            return True, 0.0
        else:
            # Quadratic penalty for larger violations
            penalty = self.penalty_weight * (violation ** 2)
            return False, penalty


class TimeWindowConstraint(Constraint):
    """
    Penalizes routes that violate delivery time windows.
    """
    
    def __init__(self, penalty_weight: float = 10.0):
        super().__init__("TimeWindowConstraint", penalty_weight)
    
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """Check time window constraints."""
        start_time = kwargs.get('start_time', 480.0)
        is_feasible, violations, total_penalty = route.check_time_window_constraint(start_time)
        
        if is_feasible:
            return True, 0.0
        else:
            # Penalty increases with number of violations and total lateness
            penalty = self.penalty_weight * (violations * 10 + total_penalty)
            return False, penalty


class RouteTimeConstraint(Constraint):
    """
    Penalizes routes that exceed maximum route time.
    """
    
    def __init__(self, penalty_weight: float = 5.0):
        super().__init__("RouteTimeConstraint", penalty_weight)
    
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """Check maximum route time constraint."""
        start_time = kwargs.get('start_time', 480.0)
        is_feasible, total_time, violation = route.check_route_time_constraint(start_time)
        
        if is_feasible:
            return True, 0.0
        else:
            # Quadratic penalty for larger time violations
            penalty = self.penalty_weight * (violation ** 2)
            return False, penalty


class TrafficConstraint(Constraint):
    """
    Penalizes routes that pass through high-traffic areas.
    """
    
    def __init__(self, traffic_zones: List[Dict] = None, penalty_weight: float = 2.0):
        super().__init__("TrafficConstraint", penalty_weight)
        self.traffic_zones = traffic_zones or []
    
    def check(self, route: Route, **kwargs) -> Tuple[bool, float]:
        """Check traffic zone violations."""
        if not self.traffic_zones:
            return True, 0.0
        
        total_traffic_penalty = 0.0
        route_coords = route.get_route_coordinates()
        
        # Check each segment for traffic zone penetration
        for i in range(len(route_coords) - 1):
            x1, y1 = route_coords[i]
            x2, y2 = route_coords[i + 1]
            
            for zone in self.traffic_zones:
                if self._segment_intersects_zone(x1, y1, x2, y2, zone):
                    slowdown_factor = zone.get('slowdown_factor', 1.5)
                    # Penalty based on slowdown factor and segment length
                    segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    penalty = (slowdown_factor - 1.0) * segment_length * self.penalty_weight
                    total_traffic_penalty += penalty
        
        return total_traffic_penalty == 0, total_traffic_penalty
    
    def _segment_intersects_zone(self, x1: float, y1: float, x2: float, y2: float, zone: Dict) -> bool:
        """Check if a line segment intersects with a traffic zone."""
        # Simple bounding box intersection check
        zone_x_min = zone['x_min']
        zone_x_max = zone['x_max']
        zone_y_min = zone['y_min']
        zone_y_max = zone['y_max']
        
        # Check if either endpoint is inside the zone
        if (zone_x_min <= x1 <= zone_x_max and zone_y_min <= y1 <= zone_y_max) or \
           (zone_x_min <= x2 <= zone_x_max and zone_y_min <= y2 <= zone_y_max):
            return True
        
        # Check for line-rectangle intersection (simplified)
        # This is a basic implementation; for production, use proper line-rectangle intersection
        return False


@dataclass
class CostEngine:
    """
    Main cost evaluation engine that combines multiple cost components and constraints.
    
    Attributes:
        cost_components: List of CostComponent instances
        constraints: List of Constraint instances
        config: Configuration dictionary
    """
    
    cost_components: List[CostComponent] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default components if none provided."""
        if not self.cost_components:
            self._initialize_default_components()
        if not self.constraints:
            self._initialize_default_constraints()
    
    def _initialize_default_components(self):
        """Initialize default cost components from config or defaults."""
        self.cost_components = [
            DistanceCostComponent(
                distance_cost_per_km=self.config.get('distance_cost_per_km', 0.5),
                weight=self.config.get('distance_weight', 1.0)
            ),
            TimeCostComponent(
                cost_per_minute=self.config.get('time_cost_per_minute', 0.25),
                weight=self.config.get('time_weight', 1.0)
            )
        ]
    
    def _initialize_default_constraints(self):
        """Initialize default constraints from config or defaults."""
        self.constraints = [
            CapacityConstraint(
                penalty_weight=self.config.get('capacity_penalty_weight', 100.0)
            ),
            TimeWindowConstraint(
                penalty_weight=self.config.get('time_window_penalty_weight', 10.0)
            ),
            RouteTimeConstraint(
                penalty_weight=self.config.get('time_penalty_weight', 5.0)
            )
        ]
        
        # Add traffic constraint if zones are provided
        traffic_zones = self.config.get('traffic_zones', [])
        if traffic_zones:
            self.constraints.append(
                TrafficConstraint(
                    traffic_zones=traffic_zones,
                    penalty_weight=self.config.get('traffic_penalty_weight', 2.0)
                )
            )
    
    def evaluate_route(self, route: Route, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a route and return comprehensive cost breakdown.
        
        Args:
            route: Route to evaluate
            **kwargs: Additional parameters (e.g., start_time)
            
        Returns:
            Dictionary with cost breakdown and feasibility information
        """
        # Calculate base costs
        base_costs = {}
        total_base_cost = 0.0
        
        for component in self.cost_components:
            cost = component.calculate(route, **kwargs)
            base_costs[component.name] = cost
            total_base_cost += cost
        
        # Check constraints and calculate penalties
        constraint_results = {}
        total_penalty = 0.0
        is_feasible = True
        
        for constraint in self.constraints:
            satisfied, penalty = constraint.check(route, **kwargs)
            constraint_results[constraint.name] = {
                'satisfied': satisfied,
                'penalty': penalty
            }
            total_penalty += penalty
            is_feasible = is_feasible and satisfied
        
        # Calculate total cost
        total_cost = total_base_cost + total_penalty
        
        # Calculate fitness (inverse of cost)
        fitness = 1.0 / (total_cost + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Get route statistics
        distance = route.calculate_total_distance()
        total_time = route.calculate_total_time()
        total_load = route.calculate_total_load()
        
        return {
            'total_cost': total_cost,
            'total_base_cost': total_base_cost,
            'total_penalty': total_penalty,
            'fitness': fitness,
            'is_feasible': is_feasible,
            'distance_km': distance,
            'total_time_min': total_time,
            'total_load_kg': total_load,
            'base_costs': base_costs,
            'constraints': constraint_results,
            'num_locations': len(route)
        }
    
    def add_cost_component(self, component: CostComponent):
        """Add a new cost component to the engine."""
        self.cost_components.append(component)
    
    def add_constraint(self, constraint: Constraint):
        """Add a new constraint to the engine."""
        self.constraints.append(constraint)
    
    def remove_cost_component(self, name: str):
        """Remove a cost component by name."""
        self.cost_components = [c for c in self.cost_components if c.name != name]
    
    def remove_constraint(self, name: str):
        """Remove a constraint by name."""
        self.constraints = [c for c in self.constraints if c.name != name]
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'CostEngine':
        """
        Create a CostEngine instance from a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configured CostEngine instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(config=config)
    
    def get_cost_breakdown(self, evaluation: Dict[str, Any]) -> str:
        """
        Format cost evaluation as a human-readable string.
        
        Args:
            evaluation: Result from evaluate_route()
            
        Returns:
            Formatted cost breakdown
        """
        output = []
        output.append("=" * 60)
        output.append("ROUTE COST EVALUATION")
        output.append("=" * 60)
        
        output.append(f"\nFEASIBILITY: {'✓ FEASIBLE' if evaluation['is_feasible'] else '✗ INFEASIBLE'}")
        output.append(f"Total cost: ${evaluation['total_cost']:.2f}")
        output.append(f"Total base cost: ${evaluation['total_base_cost']:.2f}")
        output.append(f"Total penalty: ${evaluation['total_penalty']:.2f}")
        output.append(f"Fitness: {evaluation['fitness']:.6f}")
        
        output.append(f"\nROUTE STATISTICS:")
        output.append(f"  Distance: {evaluation['distance_km']:.2f} km")
        output.append(f"  Total time: {evaluation['total_time_min']:.2f} min")
        output.append(f"  Total load: {evaluation['total_load_kg']:.2f} kg")
        output.append(f"  Locations: {evaluation['num_locations']}")
        
        output.append(f"\nBASE COSTS:")
        for name, cost in evaluation['base_costs'].items():
            output.append(f"  {name}: ${cost:.2f}")
        
        output.append(f"\nCONSTRAINTS:")
        for name, constraint in evaluation['constraints'].items():
            status = "✓" if constraint['satisfied'] else "✗"
            penalty = constraint['penalty']
            output.append(f"  {status} {name}: ${penalty:.2f}")
        
        output.append("=" * 60)
        return "\n".join(output)
    
    def __str__(self):
        """String representation of the cost engine."""
        components_str = "\n  ".join(str(c) for c in self.cost_components)
        constraints_str = "\n  ".join(f"{c.name} (penalty: {c.penalty_weight})" 
                                     for c in self.constraints)
        
        return (f"CostEngine with {len(self.cost_components)} components "
                f"and {len(self.constraints)} constraints:\n"
                f"Components:\n  {components_str}\n"
                f"Constraints:\n  {constraints_str}")