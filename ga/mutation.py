"""
Mutation operators for Genetic Algorithm.
Implements swap, inversion, and scramble mutations for permutation chromosomes.
"""

from abc import ABC, abstractmethod
from typing import List
import random
from ga.chromosome import Chromosome


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        pass


class SwapMutation(MutationOperator):
    """
    Swap mutation operator.
    Randomly swaps two genes in the chromosome.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize swap mutation.
        
        Args:
            mutation_rate: Probability of mutation per chromosome
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply swap mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Check if mutation should be performed
        if random.random() > self.mutation_rate or len(chromosome) < 2:
            return chromosome.copy()
        
        # Create copy
        mutated = chromosome.copy()
        
        # Select two distinct random positions
        pos1, pos2 = random.sample(range(len(mutated)), 2)
        
        # Swap genes
        mutated.genes[pos1], mutated.genes[pos2] = mutated.genes[pos2], mutated.genes[pos1]
        mutated.is_decoded = False
        
        return mutated


class InversionMutation(MutationOperator):
    """
    Inversion mutation operator.
    Reverses a subsequence of genes.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize inversion mutation.
        
        Args:
            mutation_rate: Probability of mutation per chromosome
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply inversion mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        if random.random() > self.mutation_rate or len(chromosome) < 3:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        length = len(mutated)
        
        # Select random subsequence
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Reverse the subsequence
        mutated.genes[start:end+1] = reversed(mutated.genes[start:end+1])
        mutated.is_decoded = False
        
        return mutated


class ScrambleMutation(MutationOperator):
    """
    Scramble mutation operator.
    Randomly shuffles a subsequence of genes.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize scramble mutation.
        
        Args:
            mutation_rate: Probability of mutation per chromosome
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply scramble mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        if random.random() > self.mutation_rate or len(chromosome) < 3:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        length = len(mutated)
        
        # Select random subsequence
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Extract subsequence, shuffle, and reinsert
        subsequence = mutated.genes[start:end+1]
        random.shuffle(subsequence)
        mutated.genes[start:end+1] = subsequence
        mutated.is_decoded = False
        
        return mutated


class InsertionMutation(MutationOperator):
    """
    Insertion mutation operator.
    Moves a gene to a different position.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize insertion mutation.
        
        Args:
            mutation_rate: Probability of mutation per chromosome
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply insertion mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        if random.random() > self.mutation_rate or len(chromosome) < 2:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        length = len(mutated)
        
        # Select gene to move and new position
        gene_pos = random.randint(0, length - 1)
        new_pos = random.randint(0, length - 1)
        
        # Ensure positions are different
        while new_pos == gene_pos:
            new_pos = random.randint(0, length - 1)
        
        # Extract gene and insert at new position
        gene = mutated.genes.pop(gene_pos)
        mutated.genes.insert(new_pos, gene)
        mutated.is_decoded = False
        
        return mutated


class AdaptiveMutation:
    """
    Adaptive mutation operator that adjusts mutation rate based on population diversity.
    """
    
    def __init__(self, base_mutation_rate: float = 0.2, 
                 min_rate: float = 0.05, max_rate: float = 0.4,
                 diversity_threshold: float = 0.3):
        """
        Initialize adaptive mutation.
        
        Args:
            base_mutation_rate: Base mutation rate
            min_rate: Minimum mutation rate
            max_rate: Maximum mutation rate
            diversity_threshold: Diversity level where mutation rate starts increasing
        """
        self.base_mutation_rate = base_mutation_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.diversity_threshold = diversity_threshold
        
        # Create mutation operators
        self.swap_mutation = SwapMutation(base_mutation_rate)
        self.inversion_mutation = InversionMutation(base_mutation_rate)
        self.insertion_mutation = InsertionMutation(base_mutation_rate)
        
        # List of mutation operators to choose from
        self.operators = [
            self.swap_mutation,
            self.inversion_mutation,
            self.insertion_mutation
        ]
    
    def update_mutation_rate(self, population_diversity: float) -> None:
        """
        Update mutation rate based on population diversity.
        
        Args:
            population_diversity: Current population diversity (0-1)
        """
        if population_diversity < self.diversity_threshold:
            # Low diversity, increase mutation rate
            increase_factor = (self.diversity_threshold - population_diversity) / self.diversity_threshold
            new_rate = min(self.max_rate, 
                          self.base_mutation_rate * (1 + increase_factor))
        else:
            # High diversity, decrease mutation rate
            new_rate = max(self.min_rate, 
                          self.base_mutation_rate * 0.8)
        
        # Update all mutation operators
        for operator in self.operators:
            if isinstance(operator, (SwapMutation, InversionMutation, InsertionMutation)):
                operator.mutation_rate = new_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply mutation using randomly selected operator.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Randomly select mutation operator
        operator = random.choice(self.operators)
        return operator.mutate(chromosome)