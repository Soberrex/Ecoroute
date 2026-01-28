"""
Adaptive genetic algorithm operators.
Dynamically adjust parameters based on population characteristics.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from collections import deque
from ga.population import Population
from ga.chromosome import Chromosome


class DiversityMonitor:
    """
    Monitors population diversity and convergence.
    
    Why it improves convergence:
    - Prevents premature convergence by detecting loss of diversity
    - Enables adaptive strategies to maintain exploration
    - Provides insights into algorithm behavior
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize diversity monitor.
        
        Args:
            window_size: Size of sliding window for trend analysis
        """
        self.window_size = window_size
        self.diversity_history = deque(maxlen=window_size)
        self.fitness_history = deque(maxlen=window_size)
    
    def update(self, population: Population) -> Dict[str, Any]:
        """
        Update monitor with current population.
        
        Args:
            population: Current population
            
        Returns:
            Diversity metrics
        """
        stats = population.get_statistics()
        
        diversity = stats.get('population_diversity', 0.0)
        avg_fitness = stats.get('avg_fitness', 0.0)
        best_fitness = stats.get('best_fitness', 0.0)
        
        self.diversity_history.append(diversity)
        self.fitness_history.append(avg_fitness)
        
        # Calculate trends
        diversity_trend = self._calculate_trend(list(self.diversity_history))
        fitness_trend = self._calculate_trend(list(self.fitness_history))
        
        # Detect stagnation
        stagnation_detected = self._detect_stagnation()
        
        # Calculate convergence metrics
        convergence_ratio = self._calculate_convergence_ratio(population)
        
        return {
            'current_diversity': diversity,
            'diversity_trend': diversity_trend,
            'fitness_trend': fitness_trend,
            'stagnation_detected': stagnation_detected,
            'convergence_ratio': convergence_ratio,
            'history_size': len(self.diversity_history)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _detect_stagnation(self) -> bool:
        """Detect if fitness improvement has stagnated."""
        if len(self.fitness_history) < self.window_size:
            return False
        
        # Check if last few generations show little improvement
        recent_improvements = []
        for i in range(1, len(self.fitness_history)):
            improvement = self.fitness_history[i] - self.fitness_history[i-1]
            recent_improvements.append(improvement)
        
        # Stagnation if average improvement is very small
        avg_improvement = np.mean(recent_improvements[-5:]) if len(recent_improvements) >= 5 else 0
        return abs(avg_improvement) < 1e-6
    
    def _calculate_convergence_ratio(self, population: Population) -> float:
        """
        Calculate how converged the population is.
        
        Returns:
            Ratio between 0 (diverse) and 1 (fully converged)
        """
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise similarity
        total_similarity = 0
        count = 0
        
        chromosomes = population.get_chromosomes()
        for i in range(len(chromosomes)):
            for j in range(i + 1, len(chromosomes)):
                similarity = self._chromosome_similarity(chromosomes[i], chromosomes[j])
                total_similarity += similarity
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_similarity / count
    
    def _chromosome_similarity(self, chrom1: Chromosome, chrom2: Chromosome) -> float:
        """Calculate similarity between two chromosomes."""
        if len(chrom1) != len(chrom2):
            return 0.0
        
        # Position-based similarity
        same_position = sum(1 for i in range(len(chrom1)) if chrom1[i] == chrom2[i])
        return same_position / len(chrom1)


class AdaptiveGA:
    """
    Adaptive Genetic Algorithm with dynamic parameter adjustment.
    
    Why it improves convergence:
    - Automatically balances exploration vs exploitation
    - Increases mutation when diversity is low
    - Adjusts selection pressure based on progress
    - Responds to stagnation with corrective measures
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize adaptive GA.
        
        Args:
            base_config: Base configuration parameters
        """
        self.base_config = base_config.copy()
        self.current_config = base_config.copy()
        self.diversity_monitor = DiversityMonitor()
        
        # Adaptive parameter ranges
        self.mutation_range = (
            base_config.get('mutation_rate_min', 0.05),
            base_config.get('mutation_rate_max', 0.4)
        )
        self.crossover_range = (
            base_config.get('crossover_rate_min', 0.6),
            base_config.get('crossover_rate_max', 0.95)
        )
        
        # State tracking
        self.generation = 0
        self.last_improvement = 0
        self.best_fitness_history = []
    
    def update_parameters(self, population: Population) -> Dict[str, Any]:
        """
        Update algorithm parameters based on population state.
        
        Args:
            population: Current population
            
        Returns:
            Updated configuration
        """
        self.generation += 1
        
        # Get current statistics
        stats = population.get_statistics()
        current_best = stats.get('best_fitness', 0.0)
        
        # Update improvement tracking
        if not self.best_fitness_history or current_best > max(self.best_fitness_history):
            self.last_improvement = self.generation
        self.best_fitness_history.append(current_best)
        
        # Get diversity metrics
        diversity_metrics = self.diversity_monitor.update(population)
        diversity = diversity_metrics['current_diversity']
        stagnation = diversity_metrics['stagnation_detected']
        
        # Calculate generations since last improvement
        gens_since_improvement = self.generation - self.last_improvement
        
        # Adjust mutation rate based on diversity
        if diversity < 0.2:  # Low diversity
            # Increase mutation rate aggressively
            mutation_rate = min(
                self.mutation_range[1],
                self.current_config.get('mutation_rate', 0.2) * 1.5
            )
        elif diversity > 0.6:  # High diversity
            # Decrease mutation rate
            mutation_rate = max(
                self.mutation_range[0],
                self.current_config.get('mutation_rate', 0.2) * 0.8
            )
        else:  # Moderate diversity
            mutation_rate = self.base_config.get('mutation_rate', 0.2)
        
        # Adjust crossover rate
        if stagnation or gens_since_improvement > 10:
            # If stagnating, increase crossover to promote exploration
            crossover_rate = min(
                self.crossover_range[1],
                self.current_config.get('crossover_rate', 0.8) * 1.1
            )
        else:
            crossover_rate = self.base_config.get('crossover_rate', 0.8)
        
        # Adjust elitism rate based on convergence
        convergence = diversity_metrics.get('convergence_ratio', 0.0)
        if convergence > 0.7:  # Highly converged
            # Reduce elitism to allow more exploration
            elitism_rate = max(0.01, self.base_config.get('elitism_rate', 0.05) * 0.5)
        else:
            elitism_rate = self.base_config.get('elitism_rate', 0.05)
        
        # Update configuration
        self.current_config.update({
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'elitism_rate': elitism_rate,
            'diversity': diversity,
            'stagnation_detected': stagnation,
            'gens_since_improvement': gens_since_improvement
        })
        
        return self.current_config
    
    def get_reset_suggestion(self) -> bool:
        """
        Check if population reset is suggested.
        
        Returns:
            True if population should be reset
        """
        if self.generation < 50:  # Don't reset too early
            return False
        
        # Reset if stagnated for too long
        if self.generation - self.last_improvement > 50:
            return True
        
        # Reset if diversity is extremely low
        diversity = self.current_config.get('diversity', 0.5)
        if diversity < 0.05:
            return True
        
        return False
    
    def get_parameter_summary(self) -> str:
        """Get summary of current adaptive parameters."""
        return (
            f"Adaptive Parameters | "
            f"Gen: {self.generation} | "
            f"Mutation: {self.current_config.get('mutation_rate', 0.2):.3f} | "
            f"Crossover: {self.current_config.get('crossover_rate', 0.8):.3f} | "
            f"Diversity: {self.current_config.get('diversity', 0.5):.3f} | "
            f"Stagnant: {self.current_config.get('stagnation_detected', False)}"
        )