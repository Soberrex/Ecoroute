"""
Selection operators for Genetic Algorithm.
Implements tournament selection and roulette wheel selection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import random
import numpy as np
from ga.chromosome import Chromosome


class SelectionOperator(ABC):
    """Abstract base class for selection operators."""
    
    @abstractmethod
    def select(self, population: List[Chromosome]) -> Chromosome:
        """
        Select a chromosome from the population.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Selected chromosome
        """
        pass


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    Selects k chromosomes at random and chooses the best one.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of chromosomes in each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[Chromosome]) -> Chromosome:
        """
        Select a chromosome using tournament selection.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Selected chromosome
        """
        # Randomly select tournament participants
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Select the best from tournament
        best = max(tournament, key=lambda x: x.fitness)
        
        return best.copy()


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel selection operator.
    Selects chromosomes with probability proportional to their fitness.
    """
    
    def __init__(self, scaling: str = 'linear'):
        """
        Initialize roulette wheel selection.
        
        Args:
            scaling: Fitness scaling method ('linear', 'exponential', or 'rank')
        """
        self.scaling = scaling
    
    def select(self, population: List[Chromosome]) -> Chromosome:
        """
        Select a chromosome using roulette wheel selection.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Selected chromosome
        """
        if not population:
            raise ValueError("Cannot select from empty population")
        
        # Scale fitness values
        scaled_fitness = self._scale_fitness([chrom.fitness for chrom in population])
        
        # Normalize to create probability distribution
        total_fitness = sum(scaled_fitness)
        
        if total_fitness <= 0:
            # If all fitnesses are zero or negative, select randomly
            return random.choice(population).copy()
        
        # Create cumulative probability distribution
        probabilities = [f / total_fitness for f in scaled_fitness]
        cumulative_probs = np.cumsum(probabilities)
        
        # Spin the roulette wheel
        r = random.random()
        
        for i, cum_prob in enumerate(cumulative_probs):
            if r <= cum_prob:
                return population[i].copy()
        
        # Fallback: return the best chromosome
        return max(population, key=lambda x: x.fitness).copy()
    
    def _scale_fitness(self, fitness_values: List[float]) -> List[float]:
        """
        Scale fitness values based on scaling method.
        
        Args:
            fitness_values: List of raw fitness values
            
        Returns:
            Scaled fitness values
        """
        if not fitness_values:
            return []
        
        if self.scaling == 'linear':
            # Shift all fitness values to be positive
            min_fitness = min(fitness_values)
            if min_fitness <= 0:
                shift = abs(min_fitness) + 1e-6
                return [f + shift for f in fitness_values]
            return fitness_values
        
        elif self.scaling == 'exponential':
            # Exponential scaling
            return [np.exp(f) for f in fitness_values]
        
        elif self.scaling == 'rank':
            # Rank-based scaling (higher rank = higher fitness)
            sorted_indices = np.argsort(fitness_values)[::-1]  # Descending
            ranks = np.zeros(len(fitness_values))
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = len(fitness_values) - rank
            return list(ranks)
        
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")


class StochasticUniversalSampling(SelectionOperator):
    """
    Stochastic Universal Sampling (SUS) selection.
    More efficient version of roulette wheel with better spread.
    """
    
    def __init__(self, num_points: int = 1):
        """
        Initialize SUS selection.
        
        Args:
            num_points: Number of equally spaced pointers
        """
        self.num_points = num_points
    
    def select(self, population: List[Chromosome]) -> List[Chromosome]:
        """
        Select multiple chromosomes using SUS.
        
        Args:
            population: List of chromosomes
            
        Returns:
            List of selected chromosomes
        """
        if not population:
            return []
        
        # Calculate total fitness
        fitness_values = [chrom.fitness for chrom in population]
        total_fitness = sum(fitness_values)
        
        if total_fitness <= 0:
            # If all fitnesses are zero or negative, select randomly
            return random.sample(population, min(self.num_points, len(population)))
        
        # Calculate point distance
        point_distance = total_fitness / self.num_points
        
        # Random start point
        start_point = random.uniform(0, point_distance)
        
        # Select chromosomes
        selected = []
        cumulative_fitness = 0
        current_index = 0
        
        for i in range(self.num_points):
            current_point = start_point + i * point_distance
            
            while cumulative_fitness < current_point and current_index < len(population):
                cumulative_fitness += fitness_values[current_index]
                current_index += 1
            
            if current_index > 0:
                selected.append(population[current_index - 1].copy())
            else:
                selected.append(population[0].copy())
        
        return selected