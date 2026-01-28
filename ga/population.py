"""
Population management for Genetic Algorithm.
Handles initialization, evaluation, and statistics of population.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from ga.chromosome import Chromosome
from domain.location import Location
from domain.vehicle import Vehicle
from cost.cost_engine import CostEngine


class Population:
    """
    Represents a population of chromosomes in the Genetic Algorithm.
    
    Attributes:
        chromosomes: List of chromosomes in the population
        size: Size of the population
        best_chromosome: Best chromosome found so far
        stats: Statistics about the population
        generation: Current generation number
    """
    
    def __init__(self, size: int = 100):
        """
        Initialize an empty population.
        
        Args:
            size: Size of the population
        """
        self.chromosomes: List[Chromosome] = []
        self.size = size
        self.best_chromosome: Optional[Chromosome] = None
        self.stats: Dict[str, Any] = {}
        self.generation = 0
    
    def initialize_random(self, locations: Dict[int, Location], depot_id: int = 0) -> None:
        """
        Initialize population with random chromosomes.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            depot_id: ID of the depot location
        """
        self.chromosomes = []
        
        for _ in range(self.size):
            chrom = Chromosome.create_random(locations, depot_id)
            self.chromosomes.append(chrom)
        
        self.generation = 0
    
    def initialize_mixed(self, locations: Dict[int, Location], depot_id: int = 0, 
                        greedy_ratio: float = 0.1) -> None:
        """
        Initialize population with mixed random and greedy chromosomes.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            depot_id: ID of the depot location
            greedy_ratio: Proportion of population to initialize with greedy algorithm
        """
        self.chromosomes = []
        
        num_greedy = int(self.size * greedy_ratio)
        num_random = self.size - num_greedy
        
        # Add greedy chromosomes
        for _ in range(num_greedy):
            chrom = Chromosome.create_greedy(locations, depot_id)
            self.chromosomes.append(chrom)
        
        # Add random chromosomes
        for _ in range(num_random):
            chrom = Chromosome.create_random(locations, depot_id)
            self.chromosomes.append(chrom)
        
        self.generation = 0
    
    def evaluate(self, locations: Dict[int, Location], vehicles: List[Vehicle],
                cost_engine: CostEngine) -> None:
        """
        Evaluate fitness of all chromosomes in the population.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            vehicles: List of available vehicles
            cost_engine: Cost engine for route evaluation
        """
        for chrom in self.chromosomes:
            chrom.evaluate_fitness(locations, vehicles, cost_engine)
        
        # Sort by fitness (descending)
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best chromosome
        if self.chromosomes:
            current_best = self.chromosomes[0]
            if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.copy()
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update population statistics."""
        if not self.chromosomes:
            self.stats = {}
            return
        
        fitnesses = [chrom.fitness for chrom in self.chromosomes]
        costs = [chrom.get_total_cost() for chrom in self.chromosomes]
        
        self.stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_cost': min(costs),
            'worst_cost': max(costs),
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'population_diversity': self.calculate_diversity(),
            'feasible_count': sum(1 for chrom in self.chromosomes 
                                 if chrom.cost_evaluation and 
                                 chrom.cost_evaluation.get('is_feasible', False))
        }
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity based on gene differences.
        
        Returns:
            Diversity metric between 0 (identical) and 1 (completely different)
        """
        if len(self.chromosomes) < 2:
            return 0.0
        
        # Compare each pair of chromosomes
        total_differences = 0
        total_comparisons = 0
        
        for i in range(len(self.chromosomes)):
            for j in range(i + 1, len(self.chromosomes)):
                chrom1 = self.chromosomes[i]
                chrom2 = self.chromosomes[j]
                
                # Count positions where genes differ
                differences = sum(1 for g1, g2 in zip(chrom1.genes, chrom2.genes) if g1 != g2)
                total_differences += differences
                total_comparisons += len(chrom1.genes)
        
        if total_comparisons == 0:
            return 0.0
        
        return total_differences / total_comparisons
    
    def get_best_chromosome(self) -> Optional[Chromosome]:
        """Get the best chromosome in the population."""
        return self.best_chromosome
    
    def get_chromosomes(self) -> List[Chromosome]:
        """Get all chromosomes in the population."""
        return self.chromosomes.copy()
    
    def set_chromosomes(self, chromosomes: List[Chromosome]) -> None:
        """
        Set the population chromosomes.
        
        Args:
            chromosomes: New list of chromosomes
        """
        if len(chromosomes) != self.size:
            raise ValueError(f"Expected {self.size} chromosomes, got {len(chromosomes)}")
        
        self.chromosomes = chromosomes.copy()
        self.generation += 1
    
    def select_elite(self, elite_ratio: float = 0.05) -> List[Chromosome]:
        """
        Select elite chromosomes (top percentage by fitness).
        
        Args:
            elite_ratio: Proportion of population to select as elite
            
        Returns:
            List of elite chromosomes
        """
        num_elite = max(1, int(self.size * elite_ratio))
        return [chrom.copy() for chrom in self.chromosomes[:num_elite]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        return self.stats.copy()
    
    def print_statistics(self) -> None:
        """Print population statistics in a readable format."""
        if not self.stats:
            print("No statistics available")
            return
        
        stats = self.stats
        print(f"\nGeneration {stats['generation']} Statistics:")
        print(f"  Best Fitness: {stats['best_fitness']:.6f}")
        print(f"  Avg Fitness:  {stats['avg_fitness']:.6f}")
        print(f"  Best Cost:    ${stats['best_cost']:.2f}")
        print(f"  Avg Cost:     ${stats['avg_cost']:.2f}")
        print(f"  Diversity:    {stats['population_diversity']:.3f}")
        print(f"  Feasible:     {stats['feasible_count']}/{self.size}")
    
    def __len__(self) -> int:
        """Number of chromosomes in the population."""
        return len(self.chromosomes)
    
    def __getitem__(self, index: int) -> Chromosome:
        """Get chromosome at specific index."""
        return self.chromosomes[index]
    
    def __str__(self) -> str:
        """String representation of the population."""
        if not self.chromosomes:
            return f"Empty Population (size: {self.size})"
        
        stats = self.stats
        return (f"Population Generation {stats.get('generation', 0)} | "
                f"Size: {self.size} | "
                f"Best: ${stats.get('best_cost', 0):.2f} | "
                f"Diversity: {stats.get('population_diversity', 0):.3f}")