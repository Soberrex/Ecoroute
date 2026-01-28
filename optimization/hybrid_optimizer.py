"""
Hybrid optimization combining Genetic Algorithm with local search.
Implements memetic algorithm approach.
"""

from typing import List, Dict, Any, Optional
import random
from ga.genetic_algorithm import GeneticAlgorithm
from ga.population import Population
from ga.chromosome import Chromosome
from optimization.local_search import LocalSearch
from optimization.adaptive_operators import AdaptiveGA, DiversityMonitor


class HybridOptimizer:
    """
    Hybrid optimizer combining GA with local search (memetic algorithm).
    
    Why it improves convergence:
    - Combines global search (GA) with local refinement (local search)
    - Faster convergence to high-quality solutions
    - Better exploitation of promising regions in search space
    - Particularly effective for complex optimization landscapes
    """
    
    def __init__(self, cost_engine, vehicles, locations, config: Dict[str, Any]):
        """
        Initialize hybrid optimizer.
        
        Args:
            cost_engine: Cost evaluation engine
            vehicles: List of vehicles
            locations: Dictionary of locations
            config: Configuration parameters
        """
        self.cost_engine = cost_engine
        self.vehicles = vehicles
        self.locations = locations
        self.config = config
        
        # Initialize components
        self.ga = GeneticAlgorithm(cost_engine, vehicles, locations, config)
        self.local_search = LocalSearch(
            use_2opt=config.get('use_2opt', True),
            use_3opt=config.get('use_3opt', False),
            intensity=config.get('local_search_intensity', 'medium')
        )
        
        self.adaptive_ga = AdaptiveGA(config)
        self.diversity_monitor = DiversityMonitor()
        
        # Hybrid-specific parameters
        self.local_search_frequency = config.get('local_search_frequency', 5)
        self.local_search_intensity = config.get('local_search_intensity', 'medium')
        self.use_adaptive = config.get('adaptive_operators', True)
    
    def optimize(self, generations: int = 100, population_size: int = 100) -> Chromosome:
        """
        Run hybrid optimization.
        
        Args:
            generations: Maximum generations
            population_size: Population size
            
        Returns:
            Best chromosome found
        """
        print("Starting Hybrid Optimization (Memetic Algorithm)")
        print("=" * 60)
        
        # Initialize GA population
        self.ga.initialize_population(population_size)
        
        best_solution = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            # Update adaptive parameters
            if self.use_adaptive:
                config_update = self.adaptive_ga.update_parameters(self.ga.population)
                
                # Update GA operators with new parameters
                self.ga.crossover_operator.crossover_rate = config_update.get('crossover_rate', 0.8)
                
                if hasattr(self.ga.mutation_operator, 'mutation_rate'):
                    self.ga.mutation_operator.mutation_rate = config_update.get('mutation_rate', 0.2)
                
                # Print adaptive parameters occasionally
                if gen % 20 == 0:
                    print(self.adaptive_ga.get_parameter_summary())
            
            # Apply local search to elite members periodically
            if gen % self.local_search_frequency == 0 and gen > 0:
                self._apply_local_search_to_elite()
            
            # Evolve one generation
            self.ga.evolve_generation()
            
            # Check for population reset
            if self.use_adaptive and self.adaptive_ga.get_reset_suggestion():
                print(f"Resetting population at generation {gen} to escape local optimum")
                self._diversify_population()
            
            # Track best solution
            current_best = self.ga.population.get_best_chromosome()
            if current_best and current_best.fitness > best_fitness:
                best_solution = current_best.copy()
                best_fitness = current_best.fitness
            
            # Print progress
            if gen % 10 == 0 or gen == 0 or gen == generations - 1:
                stats = self.ga.population.get_statistics()
                print(f"Gen {gen:3d} | "
                      f"Best: ${stats['best_cost']:.2f} | "
                      f"Avg: ${stats['avg_cost']:.2f} | "
                      f"Div: {stats['population_diversity']:.3f}")
        
        # Final local search on best solution
        if best_solution:
            print("\nApplying final intensive local search...")
            best_solution = self.local_search.optimize_chromosome(
                best_solution, self.locations, self.vehicles, self.cost_engine
            )
        
        print(f"\nHybrid optimization completed.")
        print(f"Best solution cost: ${best_solution.get_total_cost():.2f}")
        
        return best_solution
    
    def _apply_local_search_to_elite(self) -> None:
        """Apply local search to elite members of population."""
        elite_ratio = self.config.get('local_search_elite_ratio', 0.1)
        
        # Get elite chromosomes
        elite = self.ga.population.select_elite(elite_ratio)
        
        # Apply local search
        optimized_elite = []
        for chrom in elite:
            optimized = self.local_search.optimize_chromosome(
                chrom, self.locations, self.vehicles, self.cost_engine
            )
            optimized_elite.append(optimized)
        
        # Replace elite in population
        all_chromosomes = self.ga.population.get_chromosomes()
        num_elite = len(optimized_elite)
        
        # Replace first num_elite chromosomes with optimized versions
        for i in range(num_elite):
            all_chromosomes[i] = optimized_elite[i]
        
        self.ga.population.set_chromosomes(all_chromosomes)
        self.ga.population.evaluate(self.locations, self.vehicles, self.cost_engine)
    
    def _diversify_population(self) -> None:
        """Diversify population to escape local optima."""
        current_pop = self.ga.population.get_chromosomes()
        
        # Keep top 20%
        keep_count = max(1, len(current_pop) // 5)
        keep_chromosomes = current_pop[:keep_count]
        
        # Generate new diverse chromosomes
        new_chromosomes = []
        
        # Add some greedy solutions
        num_greedy = len(current_pop) // 4
        for _ in range(num_greedy):
            chrom = Chromosome.create_greedy(self.locations, self.ga.depot_id)
            new_chromosomes.append(chrom)
        
        # Add random solutions
        num_random = len(current_pop) - len(keep_chromosomes) - len(new_chromosomes)
        for _ in range(num_random):
            chrom = Chromosome.create_random(self.locations, self.ga.depot_id)
            new_chromosomes.append(chrom)
        
        # Combine kept and new chromosomes
        combined = keep_chromosomes + new_chromosomes
        
        # Ensure population size
        if len(combined) > len(current_pop):
            combined = combined[:len(current_pop)]
        elif len(combined) < len(current_pop):
            # Add more random if needed
            while len(combined) < len(current_pop):
                chrom = Chromosome.create_random(self.locations, self.ga.depot_id)
                combined.append(chrom)
        
        # Set new population
        self.ga.population.set_chromosomes(combined)
        self.ga.population.evaluate(self.locations, self.vehicles, self.cost_engine)
        
        print(f"Population diversified: kept {len(keep_chromosomes)} elite, "
              f"added {len(new_chromosomes)} new chromosomes")