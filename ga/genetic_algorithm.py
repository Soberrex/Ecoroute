"""
Main Genetic Algorithm orchestrator for EcoRoute optimization.
Coordinates selection, crossover, mutation, and evolution process.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import json

from ga.chromosome import Chromosome
from ga.population import Population
from ga.selection import TournamentSelection, RouletteWheelSelection
from ga.crossover import OrderedCrossover
from ga.mutation import SwapMutation, AdaptiveMutation
from domain.location import Location
from domain.vehicle import Vehicle
from cost.cost_engine import CostEngine


class GeneticAlgorithm:
    """
    Main Genetic Algorithm orchestrator for route optimization.
    
    Attributes:
        population: Current population
        selection_operator: Operator for parent selection
        crossover_operator: Operator for crossover
        mutation_operator: Operator for mutation
        cost_engine: Cost evaluation engine
        vehicles: List of available vehicles
        locations: Dictionary of all locations
        config: Algorithm configuration
        history: Evolution history for analysis
        best_solutions: List of best solutions over time
        start_time: Algorithm start time
    """
    
    def __init__(self, cost_engine: CostEngine, vehicles: List[Vehicle],
                 locations: Dict[int, Location], config: Dict[str, Any] = None):
        """
        Initialize Genetic Algorithm.
        
        Args:
            cost_engine: Cost evaluation engine
            vehicles: List of available vehicles
            locations: Dictionary of all locations
            config: Algorithm configuration
        """
        self.cost_engine = cost_engine
        self.vehicles = vehicles
        self.locations = locations
        self.config = config or {}
        
        # Initialize operators
        self.selection_operator = TournamentSelection(
            tournament_size=self.config.get('tournament_size', 3)
        )
        self.crossover_operator = OrderedCrossover(
            crossover_rate=self.config.get('crossover_rate', 0.8)
        )
        
        # Use adaptive mutation if configured
        if self.config.get('adaptive_mutation', True):
            self.mutation_operator = AdaptiveMutation(
                base_mutation_rate=self.config.get('mutation_rate', 0.2),
                min_rate=self.config.get('mutation_rate_min', 0.05),
                max_rate=self.config.get('mutation_rate_max', 0.4),
                diversity_threshold=self.config.get('diversity_threshold', 0.3)
            )
        else:
            self.mutation_operator = SwapMutation(
                mutation_rate=self.config.get('mutation_rate', 0.2)
            )
        
        # Initialize population
        self.population = None
        self.history: List[Dict[str, Any]] = []
        self.best_solutions: List[Chromosome] = []
        self.start_time = None
        
        # Get depot ID
        self.depot_id = next((loc_id for loc_id, loc in locations.items() if loc.is_depot), 0)
    
    def initialize_population(self, population_size: int = 100, 
                             initialization_method: str = 'mixed') -> None:
        """
        Initialize the population.
        
        Args:
            population_size: Size of the population
            initialization_method: 'random', 'greedy', or 'mixed'
        """
        self.population = Population(population_size)
        
        if initialization_method == 'random':
            self.population.initialize_random(self.locations, self.depot_id)
        elif initialization_method == 'greedy':
            # Initialize all with greedy (less diverse but good starting point)
            for _ in range(population_size):
                chrom = Chromosome.create_greedy(self.locations, self.depot_id)
                self.population.chromosomes.append(chrom)
        else:  # 'mixed' (default)
            self.population.initialize_mixed(self.locations, self.depot_id, greedy_ratio=0.1)
        
        # Evaluate initial population
        self.population.evaluate(self.locations, self.vehicles, self.cost_engine)
        self._record_generation()
    
    def evolve_generation(self) -> None:
        """
        Evolve one generation of the population.
        """
        if self.population is None:
            raise ValueError("Population not initialized. Call initialize_population first.")
        
        # Update adaptive mutation rate based on population diversity
        if isinstance(self.mutation_operator, AdaptiveMutation):
            diversity = self.population.stats.get('population_diversity', 0.5)
            self.mutation_operator.update_mutation_rate(diversity)
        
        # Create new population
        new_chromosomes = []
        
        # Apply elitism: keep best chromosomes
        elite_ratio = self.config.get('elitism_rate', 0.05)
        elite_chromosomes = self.population.select_elite(elite_ratio)
        new_chromosomes.extend(elite_chromosomes)
        
        # Fill rest of population through selection, crossover, and mutation
        while len(new_chromosomes) < self.population.size:
            # Selection
            parent1 = self.selection_operator.select(self.population.chromosomes)
            parent2 = self.selection_operator.select(self.population.chromosomes)
            
            # Crossover
            child1, child2 = self.crossover_operator.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutation_operator.mutate(child1)
            child2 = self.mutation_operator.mutate(child2)
            
            # Repair if invalid
            child1.repair(self.locations)
            child2.repair(self.locations)
            
            new_chromosomes.append(child1)
            if len(new_chromosomes) < self.population.size:
                new_chromosomes.append(child2)
        
        # Apply local optimization (2-opt) if configured
        if self.config.get('use_2opt', True):
            new_chromosomes = self._apply_local_optimization(new_chromosomes)
        
        # Set new population
        self.population.set_chromosomes(new_chromosomes)
        
        # Evaluate new population
        self.population.evaluate(self.locations, self.vehicles, self.cost_engine)
        
        # Record generation data
        self._record_generation()
    
    def _apply_local_optimization(self, chromosomes: List[Chromosome], 
                                 max_iterations: int = 10) -> List[Chromosome]:
        """
        Apply 2-opt local optimization to improve chromosomes.
        
        Args:
            chromosomes: List of chromosomes to optimize
            max_iterations: Maximum 2-opt iterations per chromosome
            
        Returns:
            Optimized chromosomes
        """
        optimized_chromosomes = []
        
        for chrom in chromosomes:
            optimized = self._two_opt(chrom, max_iterations)
            optimized_chromosomes.append(optimized)
        
        return optimized_chromosomes
    
    def _two_opt(self, chromosome: Chromosome, max_iterations: int = 10) -> Chromosome:
        """
        Apply 2-opt optimization to a chromosome.
        
        Args:
            chromosome: Chromosome to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Optimized chromosome
        """
        best_chrom = chromosome.copy()
        best_cost = best_chrom.get_total_cost() if best_chrom.is_decoded else float('inf')
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            length = len(best_chrom)
            
            for i in range(1, length - 1):
                for j in range(i + 1, length):
                    # Create new chromosome by reversing segment i..j
                    new_genes = best_chrom.genes[:]
                    new_genes[i:j+1] = reversed(new_genes[i:j+1])
                    
                    # Create new chromosome and evaluate
                    new_chrom = Chromosome(new_genes, self.depot_id)
                    new_chrom.decode(self.locations, self.vehicles, self.cost_engine)
                    new_cost = new_chrom.get_total_cost()
                    
                    # Keep if better
                    if new_cost < best_cost:
                        best_chrom = new_chrom
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            iterations += 1
        
        return best_chrom
    
    def _record_generation(self) -> None:
        """Record generation data for analysis."""
        if self.population is None:
            return
        
        stats = self.population.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        stats['elapsed_seconds'] = time.time() - self.start_time if self.start_time else 0
        
        self.history.append(stats)
        
        # Save best solution
        best_chrom = self.population.get_best_chromosome()
        if best_chrom:
            self.best_solutions.append(best_chrom.copy())
    
    def run(self, generations: int = 100, population_size: int = 100,
            early_stopping: bool = True, patience: int = 20) -> Chromosome:
        """
        Run the Genetic Algorithm for specified number of generations.
        
        Args:
            generations: Maximum number of generations
            population_size: Size of population
            early_stopping: Whether to stop early if no improvement
            patience: Number of generations to wait for improvement
            
        Returns:
            Best chromosome found
        """
        self.start_time = time.time()
        
        print(f"Starting Genetic Algorithm Optimization")
        print(f"  Generations: {generations}")
        print(f"  Population: {population_size}")
        print(f"  Locations: {len([loc for loc in self.locations.values() if not loc.is_depot])}")
        print(f"  Vehicles: {len(self.vehicles)}")
        print("=" * 60)
        
        # Initialize population
        self.initialize_population(population_size)
        
        print(f"Initial Generation:")
        self.population.print_statistics()
        
        # Evolution loop
        best_cost = self.population.stats['best_cost']
        no_improvement_count = 0
        
        for gen in range(1, generations + 1):
            # Evolve generation
            self.evolve_generation()
            
            # Check for improvement
            current_best = self.population.stats['best_cost']
            
            if current_best < best_cost - 1e-6:  # Small tolerance
                best_cost = current_best
                no_improvement_count = 0
                improvement_flag = "↑"
            else:
                no_improvement_count += 1
                improvement_flag = "→"
            
            # Print progress
            if gen % 10 == 0 or gen == 1 or gen == generations:
                stats = self.population.stats
                print(f"Gen {gen:3d} {improvement_flag} "
                      f"Best: ${stats['best_cost']:.2f} | "
                      f"Avg: ${stats['avg_cost']:.2f} | "
                      f"Div: {stats['population_diversity']:.3f} | "
                      f"Feasible: {stats['feasible_count']}/{population_size}")
            
            # Check early stopping condition
            if early_stopping and no_improvement_count >= patience:
                print(f"\nEarly stopping at generation {gen}: "
                      f"No improvement for {patience} generations")
                break
        
        # Get best solution
        best_solution = self.population.get_best_chromosome()
        
        # Print final results
        elapsed_time = time.time() - self.start_time
        print(f"\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETED")
        print(f"Total generations: {len(self.history)}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Best solution cost: ${best_solution.get_total_cost():.2f}")
        
        if best_solution.cost_evaluation:
            print(f"Number of routes: {best_solution.cost_evaluation['num_routes']}")
            print(f"Total locations: {best_solution.cost_evaluation['total_locations']}")
        
        return best_solution
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get evolution history."""
        return self.history.copy()
    
    def get_best_solutions(self) -> List[Chromosome]:
        """Get list of best solutions over generations."""
        return self.best_solutions.copy()
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save optimization results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save history as JSON
        history_file = output_path / f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert history to serializable format
        serializable_history = []
        for record in self.history:
            serializable_record = {}
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_record[key] = value.tolist()
                else:
                    serializable_record[key] = value
            serializable_history.append(serializable_record)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        # Save best solution
        best_solution = self.population.get_best_chromosome()
        if best_solution:
            solution_file = output_path / f"best_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            solution_data = {
                'genes': best_solution.genes,
                'fitness': best_solution.fitness,
                'total_cost': best_solution.get_total_cost(),
                'timestamp': datetime.now().isoformat(),
                'evaluation': best_solution.cost_evaluation
            }
            
            with open(solution_file, 'w') as f:
                json.dump(solution_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_statistics_summary(self) -> None:
        """Print summary statistics of the optimization run."""
        if not self.history:
            print("No optimization history available")
            return
        
        first_gen = self.history[0]
        last_gen = self.history[-1]
        
        improvement = first_gen['best_cost'] - last_gen['best_cost']
        improvement_percent = (improvement / first_gen['best_cost']) * 100
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Initial best cost:      ${first_gen['best_cost']:.2f}")
        print(f"Final best cost:        ${last_gen['best_cost']:.2f}")
        print(f"Improvement:            ${improvement:.2f} ({improvement_percent:.1f}%)")
        print(f"Generations:            {len(self.history)}")
        print(f"Final diversity:        {last_gen['population_diversity']:.3f}")
        print(f"Final feasible:         {last_gen['feasible_count']}/{self.population.size}")
        print(f"Time per generation:    {(self.history[-1]['elapsed_seconds'] / len(self.history)):.3f}s")