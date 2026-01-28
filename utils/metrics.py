"""
Performance metrics and evaluation utilities for EcoRoute.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from domain.route import Route
from ga.chromosome import Chromosome


class MetricsCalculator:
    """
    Calculates various performance metrics for route optimization.
    """
    
    @staticmethod
    def calculate_route_metrics(route: Route) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a single route.
        
        Args:
            route: Route to analyze
            
        Returns:
            Dictionary of route metrics
        """
        if not route.locations:
            return {}
        
        distances = []
        locations = route.locations
        
        # Calculate distances between consecutive locations
        current = route.depot
        for location in locations:
            distance = current.distance_to(location)
            distances.append(distance)
            current = location
        
        # Distance back to depot
        distances.append(current.distance_to(route.depot))
        
        # Calculate metrics
        total_distance = sum(distances)
        avg_distance = np.mean(distances) if distances else 0
        std_distance = np.std(distances) if len(distances) > 1 else 0
        
        # Calculate route efficiency (circuity ratio)
        # Ideal: straight line from depot to farthest point and back
        if locations:
            farthest_distance = max(route.depot.distance_to(loc) for loc in locations)
            ideal_distance = 2 * farthest_distance
            circuity_ratio = total_distance / ideal_distance if ideal_distance > 0 else 1.0
        else:
            circuity_ratio = 1.0
        
        # Calculate load utilization
        total_load = route.calculate_total_load()
        load_utilization = total_load / route.vehicle.max_capacity if route.vehicle.max_capacity > 0 else 0
        
        # Calculate time utilization
        total_time = route.calculate_total_time()
        time_utilization = total_time / route.vehicle.max_route_time if route.vehicle.max_route_time > 0 else 0
        
        return {
            'total_distance': total_distance,
            'avg_segment_distance': avg_distance,
            'std_segment_distance': std_distance,
            'num_locations': len(locations),
            'total_load': total_load,
            'load_utilization': load_utilization,
            'total_time': total_time,
            'time_utilization': time_utilization,
            'circuity_ratio': circuity_ratio,
            'vehicle_id': route.vehicle.id
        }
    
    @staticmethod
    def calculate_solution_metrics(chromosome: Chromosome) -> Dict[str, Any]:
        """
        Calculate metrics for a complete solution (multiple routes).
        
        Args:
            chromosome: Chromosome solution
            
        Returns:
            Dictionary of solution metrics
        """
        if not chromosome.routes:
            return {}
        
        routes = chromosome.routes
        
        # Calculate metrics for each route
        route_metrics = [MetricsCalculator.calculate_route_metrics(route) for route in routes]
        
        # Aggregate metrics
        total_distance = sum(rm.get('total_distance', 0) for rm in route_metrics)
        total_locations = sum(rm.get('num_locations', 0) for rm in route_metrics)
        
        # Calculate balance metrics
        distances = [rm.get('total_distance', 0) for rm in route_metrics]
        loads = [rm.get('total_load', 0) for rm in route_metrics]
        locations_counts = [rm.get('num_locations', 0) for rm in route_metrics]
        
        # Calculate fairness/balance metrics
        if distances:
            distance_fairness = 1 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
        else:
            distance_fairness = 1.0
        
        if loads:
            load_fairness = 1 - (np.std(loads) / np.mean(loads)) if np.mean(loads) > 0 else 0
        else:
            load_fairness = 1.0
        
        if locations_counts:
            locations_fairness = 1 - (np.std(locations_counts) / np.mean(locations_counts)) if np.mean(locations_counts) > 0 else 0
        else:
            locations_fairness = 1.0
        
        # Calculate overall utilization
        total_capacity = sum(route.vehicle.max_capacity for route in routes)
        total_max_time = sum(route.vehicle.max_route_time for route in routes)
        
        total_load = sum(loads)
        total_time = sum(rm.get('total_time', 0) for rm in route_metrics)
        
        overall_load_utilization = total_load / total_capacity if total_capacity > 0 else 0
        overall_time_utilization = total_time / total_max_time if total_max_time > 0 else 0
        
        return {
            'total_cost': chromosome.get_total_cost(),
            'fitness': chromosome.fitness,
            'num_routes': len(routes),
            'total_distance': total_distance,
            'total_locations': total_locations,
            'total_load': total_load,
            'total_time': total_time,
            'overall_load_utilization': overall_load_utilization,
            'overall_time_utilization': overall_time_utilization,
            'distance_fairness': distance_fairness,
            'load_fairness': load_fairness,
            'locations_fairness': locations_fairness,
            'route_metrics': route_metrics,
            'is_feasible': all(route.is_feasible()[0] for route in routes)
        }
    
    @staticmethod
    def calculate_improvement_metrics(initial_solution: Chromosome, 
                                    final_solution: Chromosome) -> Dict[str, Any]:
        """
        Calculate improvement metrics between two solutions.
        
        Args:
            initial_solution: Initial/benchmark solution
            final_solution: Final/optimized solution
            
        Returns:
            Dictionary of improvement metrics
        """
        initial_metrics = MetricsCalculator.calculate_solution_metrics(initial_solution)
        final_metrics = MetricsCalculator.calculate_solution_metrics(final_solution)
        
        initial_cost = initial_solution.get_total_cost()
        final_cost = final_solution.get_total_cost()
        
        if initial_cost > 0:
            cost_improvement = ((initial_cost - final_cost) / initial_cost) * 100
            cost_reduction = initial_cost - final_cost
        else:
            cost_improvement = 0.0
            cost_reduction = 0.0
        
        # Calculate distance improvement
        initial_distance = initial_metrics.get('total_distance', 0)
        final_distance = final_metrics.get('total_distance', 0)
        
        if initial_distance > 0:
            distance_improvement = ((initial_distance - final_distance) / initial_distance) * 100
        else:
            distance_improvement = 0.0
        
        # Calculate fairness improvements
        initial_fairness = initial_metrics.get('distance_fairness', 0)
        final_fairness = final_metrics.get('distance_fairness', 0)
        fairness_improvement = final_fairness - initial_fairness
        
        return {
            'cost_improvement_percent': cost_improvement,
            'cost_reduction': cost_reduction,
            'distance_improvement_percent': distance_improvement,
            'fairness_improvement': fairness_improvement,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'initial_fairness': initial_fairness,
            'final_fairness': final_fairness
        }
    
    @staticmethod
    def calculate_fuel_savings(baseline_solution: Chromosome, 
                             optimized_solution: Chromosome,
                             fuel_cost_per_liter: float = 1.5) -> Dict[str, Any]:
        """
        Calculate fuel savings between two solutions.
        
        Args:
            baseline_solution: Baseline solution
            optimized_solution: Optimized solution
            fuel_cost_per_liter: Cost of fuel per liter
            
        Returns:
            Dictionary of fuel savings metrics
        """
        def calculate_fuel_consumption(chromosome: Chromosome) -> float:
            """Calculate total fuel consumption for a solution."""
            total_fuel = 0.0
            for route in chromosome.routes:
                distance = route.calculate_total_distance()
                fuel_efficiency = route.vehicle.fuel_efficiency
                total_fuel += distance * fuel_efficiency
            return total_fuel
        
        baseline_fuel = calculate_fuel_consumption(baseline_solution)
        optimized_fuel = calculate_fuel_consumption(optimized_solution)
        
        fuel_savings = baseline_fuel - optimized_fuel
        
        if baseline_fuel > 0:
            fuel_savings_percent = (fuel_savings / baseline_fuel) * 100
        else:
            fuel_savings_percent = 0.0
        
        fuel_cost_savings = fuel_savings * fuel_cost_per_liter
        
        return {
            'baseline_fuel_liters': baseline_fuel,
            'optimized_fuel_liters': optimized_fuel,
            'fuel_savings_liters': fuel_savings,
            'fuel_savings_percent': fuel_savings_percent,
            'fuel_cost_savings': fuel_cost_savings,
            'daily_savings': fuel_cost_savings,
            'monthly_savings': fuel_cost_savings * 22,  # Assuming 22 working days
            'yearly_savings': fuel_cost_savings * 22 * 12
        }


class PerformanceMonitor:
    """
    Monitors optimization performance over time.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_history = []
        self.start_time = datetime.now()
    
    def record_generation(self, generation: int, population_stats: Dict[str, Any],
                         best_solution: Chromosome) -> None:
        """
        Record performance metrics for a generation.
        
        Args:
            generation: Generation number
            population_stats: Population statistics
            best_solution: Best solution in this generation
        """
        solution_metrics = MetricsCalculator.calculate_solution_metrics(best_solution)
        
        record = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': (datetime.now() - self.start_time).total_seconds(),
            'population_stats': population_stats,
            'solution_metrics': solution_metrics
        }
        
        self.metrics_history.append(record)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        first_gen = self.metrics_history[0]
        last_gen = self.metrics_history[-1]
        
        # Calculate improvements
        initial_cost = first_gen['solution_metrics']['total_cost']
        final_cost = last_gen['solution_metrics']['total_cost']
        
        if initial_cost > 0:
            improvement_percent = ((initial_cost - final_cost) / initial_cost) * 100
        else:
            improvement_percent = 0.0
        
        # Calculate convergence speed
        convergence_gen = None
        target_percent = 0.95  # 95% of final improvement
        target_cost = initial_cost - (initial_cost - final_cost) * target_percent
        
        for record in self.metrics_history:
            if record['solution_metrics']['total_cost'] <= target_cost:
                convergence_gen = record['generation']
                break
        
        return {
            'total_generations': len(self.metrics_history),
            'total_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'improvement_percent': improvement_percent,
            'convergence_generation': convergence_gen,
            'time_per_generation': ((datetime.now() - self.start_time).total_seconds() / 
                                   len(self.metrics_history))
        }
    
    def export_metrics(self, filename: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert to serializable format
        serializable_history = []
        for record in self.metrics_history:
            serializable_record = {}
            for key, value in record.items():
                if isinstance(value, datetime):
                    serializable_record[key] = value.isoformat()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_record[key] = value.tolist()
                else:
                    serializable_record[key] = value
            serializable_history.append(serializable_record)
        
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=2)