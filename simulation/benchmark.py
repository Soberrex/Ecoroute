"""
Benchmarking utilities for comparing EcoRoute against baseline algorithms.
"""

import random
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt  # Added missing import

from domain.location import Location  # Fixed import path
from domain.vehicle import Vehicle  # Fixed import path
from domain.route import Route  # Fixed import path
from cost.cost_engine import CostEngine
from ga.chromosome import Chromosome
from ga.genetic_algorithm import GeneticAlgorithm
from optimization.hybrid_optimizer import HybridOptimizer
from visualization.dashboard import ConvergencePlotter


class RandomRouter:
    """
    Baseline: Random route generator.
    """
    
    def __init__(self, cost_engine: CostEngine):
        """
        Initialize random router.
        
        Args:
            cost_engine: Cost evaluation engine
        """
        self.cost_engine = cost_engine
        self.name = "Random Router"
    
    def solve(self, locations: Dict[int, Location], vehicles: List[Vehicle],
              num_solutions: int = 100) -> List[Chromosome]:
        """
        Generate random solutions.
        
        Args:
            locations: Dictionary of locations
            vehicles: List of vehicles
            num_solutions: Number of random solutions to generate
            
        Returns:
            List of random chromosomes
        """
        solutions = []
        depot_id = next((loc_id for loc_id, loc in locations.items() if loc.is_depot), 0)
        
        for _ in range(num_solutions):
            chrom = Chromosome.create_random(locations, depot_id)
            chrom.decode(locations, vehicles, self.cost_engine)
            solutions.append(chrom)
        
        # Sort by cost (ascending)
        solutions.sort(key=lambda x: x.get_total_cost())
        
        return solutions
    
    def get_best_solution(self, locations: Dict[int, Location], 
                         vehicles: List[Vehicle], num_trials: int = 100) -> Chromosome:
        """
        Get best random solution after multiple trials.
        
        Args:
            locations: Dictionary of locations
            vehicles: List of vehicles
            num_trials: Number of random trials
            
        Returns:
            Best random solution found
        """
        solutions = self.solve(locations, vehicles, num_trials)
        return solutions[0] if solutions else None


class GreedyRouter:
    """
    Baseline: Nearest-neighbor greedy algorithm.
    """
    
    def __init__(self, cost_engine: CostEngine):
        """
        Initialize greedy router.
        
        Args:
            cost_engine: Cost evaluation engine
        """
        self.cost_engine = cost_engine
        self.name = "Greedy Router"
    
    def solve(self, locations: Dict[int, Location], vehicles: List[Vehicle]) -> Chromosome:
        """
        Solve using nearest-neighbor greedy algorithm.
        
        Args:
            locations: Dictionary of locations
            vehicles: List of vehicles
            
        Returns:
            Greedy solution chromosome
        """
        depot_id = next((loc_id for loc_id, loc in locations.items() if loc.is_depot), 0)
        
        # Create greedy chromosome
        chrom = Chromosome.create_greedy(locations, depot_id)
        chrom.decode(locations, vehicles, self.cost_engine)
        
        return chrom
    
    def solve_multiple(self, locations: Dict[int, Location], vehicles: List[Vehicle],
                      num_starts: int = 10) -> List[Chromosome]:
        """
        Solve multiple times with different starting points.
        
        Args:
            locations: Dictionary of locations
            vehicles: List of vehicles
            num_starts: Number of different starting points
            
        Returns:
            List of greedy solutions
        """
        solutions = []
        depot_id = next((loc_id for loc_id, loc in locations.items() if loc.is_depot), 0)
        
        # Get all non-depot locations
        non_depot_ids = [loc_id for loc_id, loc in locations.items() if not loc.is_depot]
        
        if len(non_depot_ids) < num_starts:
            num_starts = len(non_depot_ids)
        
        # Try different starting locations
        start_indices = random.sample(range(len(non_depot_ids)), num_starts)
        
        for start_idx in start_indices:
            # Reorder starting from different point
            reordered_ids = non_depot_ids[start_idx:] + non_depot_ids[:start_idx]
            
            # Create greedy route from this starting point
            route = []
            unvisited = reordered_ids.copy()
            current_id = unvisited.pop(0)
            route.append(current_id)
            
            while unvisited:
                current_loc = locations[current_id]
                
                # Find nearest unvisited
                nearest_id = min(unvisited,
                               key=lambda loc_id: current_loc.distance_to(locations[loc_id]))
                
                route.append(nearest_id)
                unvisited.remove(nearest_id)
                current_id = nearest_id
            
            # Create and evaluate chromosome
            chrom = Chromosome(route, depot_id)
            chrom.decode(locations, vehicles, self.cost_engine)
            solutions.append(chrom)
        
        # Sort by cost
        solutions.sort(key=lambda x: x.get_total_cost())
        
        return solutions


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for comparing algorithms.
    """
    
    def __init__(self, cost_engine: CostEngine, vehicles: List[Vehicle],
                 locations: Dict[int, Location]):
        """
        Initialize benchmark suite.
        
        Args:
            cost_engine: Cost evaluation engine
            vehicles: List of vehicles
            locations: Dictionary of locations
        """
        self.cost_engine = cost_engine
        self.vehicles = vehicles
        self.locations = locations
        
        # Initialize algorithms
        self.random_router = RandomRouter(cost_engine)
        self.greedy_router = GreedyRouter(cost_engine)
        
        # Results storage
        self.results = {}
        self.comparison_data = {}
    
    def run_benchmark(self, ga_config: Dict[str, Any] = None, 
                     num_trials: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive benchmark of all algorithms.
        
        Args:
            ga_config: Genetic Algorithm configuration
            num_trials: Number of trials for each algorithm
            
        Returns:
            Benchmark results
        """
        print("=" * 60)
        print("RUNNING ECOROUTE BENCHMARK SUITE")
        print("=" * 60)
        
        ga_config = ga_config or {
            'population_size': 50,
            'generations': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elitism_rate': 0.05,
            'tournament_size': 3,
            'adaptive_mutation': True,
            'use_2opt': True
        }
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'num_locations': len([loc for loc in self.locations.values() if not loc.is_depot]),
            'num_vehicles': len(self.vehicles),
            'trials': num_trials
        }
        
        # 1. Random Router Benchmark
        print("\n1. Benchmarking Random Router...")
        random_results = self._benchmark_random(num_trials)
        benchmark_results['random'] = random_results
        self._print_algorithm_results("Random Router", random_results)
        
        # 2. Greedy Router Benchmark
        print("\n2. Benchmarking Greedy Router...")
        greedy_results = self._benchmark_greedy(num_trials)
        benchmark_results['greedy'] = greedy_results
        self._print_algorithm_results("Greedy Router", greedy_results)
        
        # 3. Standard GA Benchmark
        print("\n3. Benchmarking Standard Genetic Algorithm...")
        ga_results = self._benchmark_ga(ga_config, num_trials, use_hybrid=False)
        benchmark_results['genetic_algorithm'] = ga_results
        self._print_algorithm_results("Genetic Algorithm", ga_results)
        
        # 4. Hybrid GA Benchmark
        print("\n4. Benchmarking Hybrid Genetic Algorithm...")
        hybrid_results = self._benchmark_ga(ga_config, num_trials, use_hybrid=True)
        benchmark_results['hybrid_ga'] = hybrid_results
        self._print_algorithm_results("Hybrid GA", hybrid_results)
        
        # 5. Comparative Analysis
        print("\n5. Performing Comparative Analysis...")
        comparison = self._compare_algorithms(benchmark_results)
        benchmark_results['comparison'] = comparison
        
        # Print final comparison
        self._print_comparison_table(benchmark_results)
        
        # Save results
        self.results = benchmark_results
        self._save_results(benchmark_results)
        
        return benchmark_results
    
    def _benchmark_random(self, num_trials: int) -> Dict[str, Any]:
        """Benchmark random router."""
        start_time = time.time()
        
        # Generate multiple random solutions
        solutions = self.random_router.solve(self.locations, self.vehicles, num_trials * 10)
        
        # Get statistics
        costs = [chrom.get_total_cost() for chrom in solutions[:num_trials]]
        
        return {
            'best_cost': min(costs) if costs else float('inf'),
            'worst_cost': max(costs) if costs else 0.0,
            'avg_cost': np.mean(costs) if costs else 0.0,
            'std_cost': np.std(costs) if costs else 0.0,
            'execution_time': time.time() - start_time,
            'num_solutions': len(costs)
        }
    
    def _benchmark_greedy(self, num_trials: int) -> Dict[str, Any]:
        """Benchmark greedy router."""
        start_time = time.time()
        
        # Generate multiple greedy solutions with different starts
        solutions = self.greedy_router.solve_multiple(self.locations, self.vehicles, num_trials)
        
        costs = [chrom.get_total_cost() for chrom in solutions]
        
        return {
            'best_cost': min(costs) if costs else float('inf'),
            'worst_cost': max(costs) if costs else 0.0,
            'avg_cost': np.mean(costs) if costs else 0.0,
            'std_cost': np.std(costs) if costs else 0.0,
            'execution_time': time.time() - start_time,
            'num_solutions': len(costs)
        }
    
    def _benchmark_ga(self, config: Dict[str, Any], num_trials: int, 
                     use_hybrid: bool = False) -> Dict[str, Any]:
        """Benchmark Genetic Algorithm."""
        all_costs = []
        all_times = []
        all_histories = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...")
            
            start_time = time.time()
            
            if use_hybrid:
                optimizer = HybridOptimizer(self.cost_engine, self.vehicles, 
                                          self.locations, config)
                best_solution = optimizer.optimize(
                    generations=config.get('generations', 50),
                    population_size=config.get('population_size', 50)
                )
                history = optimizer.ga.get_history()
            else:
                ga = GeneticAlgorithm(self.cost_engine, self.vehicles, 
                                    self.locations, config)
                best_solution = ga.run(
                    generations=config.get('generations', 50),
                    population_size=config.get('population_size', 50),
                    early_stopping=False
                )
                history = ga.get_history()
            
            execution_time = time.time() - start_time
            
            if best_solution:
                all_costs.append(best_solution.get_total_cost())
                all_times.append(execution_time)
                all_histories.append(history)
        
        return {
            'best_cost': min(all_costs) if all_costs else float('inf'),
            'worst_cost': max(all_costs) if all_costs else 0.0,
            'avg_cost': np.mean(all_costs) if all_costs else 0.0,
            'std_cost': np.std(all_costs) if all_costs else 0.0,
            'avg_execution_time': np.mean(all_times) if all_times else 0.0,
            'std_execution_time': np.std(all_times) if all_times else 0.0,
            'num_trials': num_trials,
            'histories': all_histories[:2]  # Keep only first 2 for visualization
        }
    
    def _compare_algorithms(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare algorithm performance."""
        comparison = {}
        
        algorithms = ['random', 'greedy', 'genetic_algorithm', 'hybrid_ga']
        algorithm_names = {
            'random': 'Random Router',
            'greedy': 'Greedy Router',
            'genetic_algorithm': 'Genetic Algorithm',
            'hybrid_ga': 'Hybrid GA'
        }
        
        # Calculate improvements over baseline (random)
        baseline_cost = results.get('random', {}).get('best_cost', float('inf'))
        
        for algo in algorithms:
            if algo in results:
                algo_result = results[algo]
                best_cost = algo_result.get('best_cost', float('inf'))
                
                if baseline_cost > 0 and best_cost < float('inf'):
                    improvement = ((baseline_cost - best_cost) / baseline_cost) * 100
                else:
                    improvement = 0.0
                
                comparison[algo] = {
                    'name': algorithm_names[algo],
                    'best_cost': best_cost,
                    'improvement_over_random': improvement,
                    'execution_time': algo_result.get('avg_execution_time', 
                                                    algo_result.get('execution_time', 0)),
                    'cost_reliability': 1.0 - (algo_result.get('std_cost', 0) / 
                                             max(1.0, algo_result.get('avg_cost', 1.0)))
                }
        
        # Rank algorithms by cost
        ranked = sorted(comparison.items(), 
                       key=lambda x: x[1]['best_cost'])
        
        comparison['ranking'] = [algo[0] for algo in ranked]
        
        return comparison
    
    def _print_algorithm_results(self, algorithm_name: str, results: Dict[str, Any]) -> None:
        """Print algorithm results in a formatted way."""
        print(f"  {algorithm_name}:")
        print(f"    Best Cost:    ${results.get('best_cost', 0):.2f}")
        print(f"    Average Cost: ${results.get('avg_cost', 0):.2f}")
        print(f"    Std Dev:      ${results.get('std_cost', 0):.2f}")
        print(f"    Time:         {results.get('execution_time', results.get('avg_execution_time', 0)):.2f}s")
    
    def _print_comparison_table(self, results: Dict[str, Any]) -> None:
        """Print comparison table of all algorithms."""
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Algorithm':<20} {'Best Cost':<15} {'Improvement':<15} {'Time (s)':<15} {'Reliability':<15}")
        print("-" * 80)
        
        comparison = results.get('comparison', {})
        
        for algo_key, algo_data in comparison.items():
            if algo_key == 'ranking':
                continue
            
            name = algo_data.get('name', algo_key)
            best_cost = algo_data.get('best_cost', 0)
            improvement = algo_data.get('improvement_over_random', 0)
            exec_time = algo_data.get('execution_time', 0)
            reliability = algo_data.get('cost_reliability', 0)
            
            print(f"{name:<20} ${best_cost:<14.2f} {improvement:<14.1f}% {exec_time:<14.2f} {reliability:<14.3f}")
        
        print("=" * 80)
        
        # Print recommendation
        ranking = comparison.get('ranking', [])
        if ranking:
            best_algo = comparison[ranking[0]]
            print(f"\nRECOMMENDATION: Use {best_algo['name']}")
            print(f"  - Achieves ${best_algo['best_cost']:.2f} cost")
            print(f"  - {best_algo['improvement_over_random']:.1f}% better than random")
            print(f"  - {best_algo['cost_reliability']:.1%} cost reliability")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        results_file = output_dir / f"benchmark_{timestamp}.json"
        
        # Convert to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(results, output_dir, timestamp)
        
        print(f"\nBenchmark results saved to {results_file}")
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def _generate_visualizations(self, results: Dict[str, Any], 
                               output_dir: Path, timestamp: str) -> None:
        """Generate benchmark visualizations."""
        plotter = ConvergencePlotter()
        
        # Prepare histories for comparison
        histories = {}
        
        if 'genetic_algorithm' in results and 'histories' in results['genetic_algorithm']:
            ga_histories = results['genetic_algorithm']['histories']
            if ga_histories:
                # Use first trial history
                histories['Genetic Algorithm'] = ga_histories[0]
        
        if 'hybrid_ga' in results and 'histories' in results['hybrid_ga']:
            hybrid_histories = results['hybrid_ga']['histories']
            if hybrid_histories:
                histories['Hybrid GA'] = hybrid_histories[0]
        
        # Plot convergence comparison
        if len(histories) >= 2:
            try:
                fig = plotter.plot_comparison(
                    histories,
                    title=f"Algorithm Convergence Comparison ({timestamp})"
                )
                fig.savefig(output_dir / f"convergence_comparison_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate convergence plot: {e}")
        
        # Create cost comparison bar chart
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            algorithms = []
            best_costs = []
            
            for algo_key in ['random', 'greedy', 'genetic_algorithm', 'hybrid_ga']:
                if algo_key in results:
                    algo_name = {
                        'random': 'Random',
                        'greedy': 'Greedy',
                        'genetic_algorithm': 'GA',
                        'hybrid_ga': 'Hybrid GA'
                    }[algo_key]
                    
                    algorithms.append(algo_name)
                    best_costs.append(results[algo_key].get('best_cost', 0))
            
            bars = ax.bar(algorithms, best_costs, color=['red', 'orange', 'blue', 'green'])
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Best Cost ($)')
            ax.set_title(f'Algorithm Performance Comparison ({timestamp})')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, cost in zip(bars, best_costs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${cost:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            fig.savefig(output_dir / f"cost_comparison_{timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not generate cost comparison chart: {e}")