#!/usr/bin/env python3
"""
EcoRoute: Evolutionary Logistics Optimization Engine
Main entry point for the application.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from domain.location import Location
from domain.vehicle import Vehicle
from domain.route import Route
from cost.cost_engine import CostEngine
from ga.genetic_algorithm import GeneticAlgorithm
from optimization.hybrid_optimizer import HybridOptimizer
from simulation.benchmark import BenchmarkSuite
from visualization.dashboard import Dashboard, RouteVisualizer, ConvergencePlotter
from utils.metrics import MetricsCalculator, PerformanceMonitor


class EcoRouteOptimizer:
    """
    Main orchestrator for EcoRoute optimization engine.
    """
    
    def __init__(self, config_path: str = "ecoroute/config/settings.json"):
        """
        Initialize EcoRoute optimizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Core components
        self.locations: Dict[int, Location] = {}
        self.vehicles: List[Vehicle] = []
        self.cost_engine: Optional[CostEngine] = None
        self.traffic_simulator = None
        
        # Results
        self.best_solution = None
        self.optimization_history = []
        self.performance_monitor = PerformanceMonitor()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_path} not found. Using defaults.")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "genetic_algorithm": {
                "population_size": 100,
                "generations": 200,
                "crossover_rate": 0.8,
                "mutation_rate": 0.2,
                "elitism_rate": 0.05,
                "tournament_size": 5,
                "adaptive_mutation": True,
                "use_2opt": True
            },
            "vehicles": {
                "default_capacity": 100.0,
                "default_fuel_efficiency": 0.2,
                "default_max_route_time": 480.0
            }
        }
    
    def generate_sample_locations(self, num_customers: int = 50, 
                                 area_size: float = 100.0) -> None:
        """
        Generate sample locations for testing.
        
        Args:
            num_customers: Number of customer locations to generate
            area_size: Size of the area (square)
        """
        import random
        
        self.locations = {}
        
        # Create depot
        depot = Location(
            id=0,
            x=area_size / 2,
            y=area_size / 2,
            name="Central Depot",
            is_depot=True
        )
        self.locations[0] = depot
        
        # Create customers
        for i in range(1, num_customers + 1):
            location = Location(
                id=i,
                x=random.uniform(0, area_size),
                y=random.uniform(0, area_size),
                demand_weight=random.uniform(10, 50),
                service_time=random.uniform(3, 15),
                name=f"Customer {i}",
                time_window_start=random.uniform(480, 720),
                time_window_end=random.uniform(840, 1020)
            )
            self.locations[i] = location
        
        print(f"Generated {num_customers} sample locations in {area_size}x{area_size} area")
    
    def setup_vehicles(self, num_vehicles: int = 3) -> None:
        """
        Set up delivery vehicles.
        
        Args:
            num_vehicles: Number of vehicles to create
        """
        vehicle_config = self.config.get('vehicles', {})
        
        self.vehicles = []
        
        for i in range(num_vehicles):
            vehicle = Vehicle(
                max_capacity=vehicle_config.get('default_capacity', 100.0),
                fuel_efficiency=vehicle_config.get('default_fuel_efficiency', 0.2),
                max_route_time=vehicle_config.get('default_max_route_time', 480.0),
                fixed_cost=vehicle_config.get('default_fixed_cost', 50.0),
                variable_cost=vehicle_config.get('default_variable_cost', 0.5),
                fuel_cost_per_liter=vehicle_config.get('fuel_cost_per_liter', 1.5)
            )
            self.vehicles.append(vehicle)
        
        print(f"Set up {num_vehicles} vehicles")
    
    def setup_cost_engine(self) -> None:
        """Set up the cost evaluation engine."""
        self.cost_engine = CostEngine(config=self.config)
        print("Cost engine initialized")
    
    def optimize(self, algorithm: str = 'hybrid', **kwargs) -> Any:
        """
        Run optimization with specified algorithm.
        
        Args:
            algorithm: Optimization algorithm ('genetic', 'hybrid', or 'benchmark')
            **kwargs: Additional parameters for optimization
            
        Returns:
            Best solution found
        """
        if not self.locations:
            print("Error: No locations loaded. Use generate_sample_locations() first.")
            return None
        
        if not self.vehicles:
            print("Warning: No vehicles set up. Using default vehicles.")
            self.setup_vehicles()
        
        if not self.cost_engine:
            self.setup_cost_engine()
        
        ga_config = self.config.get('genetic_algorithm', {})
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if key in ga_config:
                ga_config[key] = value
        
        print(f"\nStarting optimization with {algorithm} algorithm...")
        
        if algorithm == 'genetic':
            optimizer = GeneticAlgorithm(
                self.cost_engine, self.vehicles, self.locations, ga_config
            )
            
            best_solution = optimizer.run(
                generations=ga_config.get('generations', 200),
                population_size=ga_config.get('population_size', 100),
                early_stopping=ga_config.get('early_stopping', True)
            )
            
            self.optimization_history = optimizer.get_history()
        
        elif algorithm == 'hybrid':
            optimizer = HybridOptimizer(
                self.cost_engine, self.vehicles, self.locations, ga_config
            )
            
            best_solution = optimizer.optimize(
                generations=ga_config.get('generations', 200),
                population_size=ga_config.get('population_size', 100)
            )
            
            self.optimization_history = optimizer.ga.get_history()
        
        elif algorithm == 'benchmark':
            benchmark = BenchmarkSuite(self.cost_engine, self.vehicles, self.locations)
            results = benchmark.run_benchmark(ga_config, num_trials=3)
            best_solution = None
        
        else:
            print(f"Error: Unknown algorithm '{algorithm}'")
            return None
        
        self.best_solution = best_solution
        
        if best_solution:
            print(f"\nOptimization completed successfully!")
            print(f"Best solution cost: ${best_solution.get_total_cost():.2f}")
        
        return best_solution
    
    def visualize_solution(self, solution: Any = None, save_path: str = None) -> None:
        """
        Visualize the solution.
        
        Args:
            solution: Solution to visualize (uses best_solution if None)
            save_path: Path to save visualization
        """
        if solution is None:
            solution = self.best_solution
        
        if solution is None:
            print("Error: No solution to visualize.")
            return
        
        visualizer = RouteVisualizer()
        
        if solution.routes:
            fig = visualizer.plot_chromosome(
                solution,
                title="EcoRoute Optimized Solution"
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.show()
        else:
            print("Error: Solution has no decoded routes.")


def main():
    """Main entry point for EcoRoute CLI."""
    parser = argparse.ArgumentParser(
        description="EcoRoute: Evolutionary Logistics Optimization Engine"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="ecoroute/config/settings.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--generate",
        type=int,
        help="Generate sample locations with N customers"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=['genetic', 'hybrid', 'benchmark'],
        default='hybrid',
        help="Optimization algorithm to use"
    )
    
    parser.add_argument(
        "--vehicles",
        type=int,
        default=3,
        help="Number of vehicles"
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations for GA"
    )
    
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size for GA"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results after optimization"
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = EcoRouteOptimizer(args.config)
    
    # Generate locations
    if args.generate:
        optimizer.generate_sample_locations(args.generate)
    else:
        print("No locations specified. Generating sample data...")
        optimizer.generate_sample_locations(20)
    
    # Setup vehicles
    optimizer.setup_vehicles(args.vehicles)
    
    # Setup cost engine
    optimizer.setup_cost_engine()
    
    # Run optimization
    best_solution = optimizer.optimize(
        algorithm=args.algorithm,
        generations=args.generations,
        population_size=args.population
    )
    
    # Visualize if requested
    if args.visualize and best_solution:
        optimizer.visualize_solution(best_solution)


if __name__ == "__main__":
    main()
