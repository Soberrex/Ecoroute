#!/usr/bin/env python3
"""
Test script for EcoRoute Cost Engine.
"""

import json
import tempfile
from domain.location import Location
from domain.vehicle import Vehicle
from domain.route import Route
from cost.cost_engine import CostEngine, DistanceCostComponent, TimeCostComponent


def test_cost_engine():
    """Test basic functionality of the Cost Engine."""
    
    print("Testing EcoRoute Cost Engine...")
    print("=" * 60)
    
    # Create test data
    depot = Location(id=0, x=0, y=0, is_depot=True)
    
    locations = [
        Location(id=1, x=10, y=10, demand_weight=25, time_window_start=480, time_window_end=720),
        Location(id=2, x=30, y=20, demand_weight=15, time_window_start=540, time_window_end=780),
        Location(id=3, x=20, y=40, demand_weight=30, time_window_start=600, time_window_end=840),
    ]
    
    vehicle = Vehicle(
        max_capacity=50,  # Deliberately small to test capacity constraint
        fuel_efficiency=0.2,
        max_route_time=300,
        fixed_cost=50,
        variable_cost=0.5
    )
    
    route = Route(vehicle=vehicle, depot=depot, speed_kmh=40)
    for loc in locations:
        route.add_location(loc)
    
    print(f"Test route: {route}")
    
    # Create cost engine with custom components
    cost_engine = CostEngine()
    
    # Add specific components
    cost_engine.cost_components = [
        DistanceCostComponent(distance_cost_per_km=0.5, weight=1.0),
        TimeCostComponent(cost_per_minute=0.25, weight=1.0)
    ]
    
    print(f"\nCost Engine Configuration:")
    print(cost_engine)
    
    # Evaluate route
    evaluation = cost_engine.evaluate_route(route, start_time=480)
    
    print("\nRoute Evaluation:")
    print(cost_engine.get_cost_breakdown(evaluation))
    
    # Test with configuration file
    print("\n" + "=" * 60)
    print("Testing Cost Engine from Configuration File...")
    
    # Create temporary config file
    config_data = {
        "distance_cost_per_km": 0.6,
        "time_cost_per_minute": 0.3,
        "distance_weight": 1.2,
        "time_weight": 1.0,
        "capacity_penalty_weight": 150.0,
        "time_window_penalty_weight": 15.0,
        "time_penalty_weight": 8.0,
        "traffic_zones": [
            {"x_min": 0, "x_max": 15, "y_min": 0, "y_max": 15, "slowdown_factor": 1.5}
        ],
        "traffic_penalty_weight": 3.0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Create cost engine from config file
        configured_engine = CostEngine.from_config_file(config_path)
        
        print(f"\nConfigured Cost Engine:")
        print(configured_engine)
        
        # Evaluate with configured engine
        evaluation2 = configured_engine.evaluate_route(route, start_time=480)
        
        print("\nRoute Evaluation (Configured Engine):")
        print(configured_engine.get_cost_breakdown(evaluation2))
        
    finally:
        # Clean up temp file
        import os
        os.unlink(config_path)
    
    # Test adding custom component
    print("\n" + "=" * 60)
    print("Testing Custom Cost Component...")
    
    from cost.cost_engine import CostComponent
    
    class CustomServiceCost(CostComponent):
        """Custom cost component for service time."""
        
        def __init__(self, cost_per_minute: float = 0.1, weight: float = 0.5):
            super().__init__("ServiceTimeCost", weight)
            self.cost_per_minute = cost_per_minute
        
        def calculate(self, route: Route, **kwargs) -> float:
            total_service_time = sum(loc.service_time for loc in route.locations)
            return total_service_time * self.cost_per_minute * self.weight
    
    # Add custom component
    custom_engine = CostEngine()
    custom_engine.add_cost_component(CustomServiceCost(cost_per_minute=0.1))
    
    print(f"\nCustom Engine with Service Time Cost:")
    print(custom_engine)
    
    evaluation3 = custom_engine.evaluate_route(route, start_time=480)
    print("\nRoute Evaluation (Custom Engine):")
    print(custom_engine.get_cost_breakdown(evaluation3))
    
    print("\nCost Engine test completed successfully!")


if __name__ == "__main__":
    test_cost_engine()