"""
Crossover operators for Genetic Algorithm.
Implements Ordered Crossover (OX) for permutation-based chromosomes.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import random
from ga.chromosome import Chromosome


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parent chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two child chromosomes
        """
        pass


class OrderedCrossover(CrossoverOperator):
    """
    Ordered Crossover (OX) operator for permutation problems.
    Preserves relative order of genes from parents.
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize Ordered Crossover.
        
        Args:
            crossover_rate: Probability of performing crossover
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform Ordered Crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two child chromosomes
        """
        # Check if crossover should be performed
        if random.random() > self.crossover_rate or len(parent1) < 3:
            # Return copies of parents (no crossover)
            return parent1.copy(), parent2.copy()
        
        # Select two random crossover points
        length = len(parent1)
        point1 = random.randint(0, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        # Create children
        child1 = self._ox_crossover(parent1, parent2, point1, point2)
        child2 = self._ox_crossover(parent2, parent1, point1, point2)
        
        return child1, child2
    
    def _ox_crossover(self, parent1: Chromosome, parent2: Chromosome, 
                     start: int, end: int) -> Chromosome:
        """
        Perform OX crossover from parent1 to parent2.
        
        Args:
            parent1: Source parent for middle section
            parent2: Source parent for remaining genes
            start: Start index of middle section
            end: End index of middle section
            
        Returns:
            Child chromosome
        """
        length = len(parent1)
        
        # Initialize child with None values
        child_genes = [None] * length
        
        # Copy middle section from parent1
        for i in range(start, end + 1):
            child_genes[i] = parent1[i]
        
        # Fill remaining positions from parent2
        parent2_index = 0
        child_index = 0
        
        while child_index < length:
            # Skip already filled positions
            if child_index == start:
                child_index = end + 1
                if child_index >= length:
                    break
            
            # Get next gene from parent2
            gene = parent2[parent2_index]
            parent2_index += 1
            
            # If gene not already in child, add it
            if gene not in child_genes:
                child_genes[child_index] = gene
                child_index += 1
        
        return Chromosome(child_genes, parent1.depot_id)


class PartiallyMappedCrossover(CrossoverOperator):
    """
    Partially Mapped Crossover (PMX) operator.
    Creates children by combining sections of parents with mapping relations.
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize PMX.
        
        Args:
            crossover_rate: Probability of performing crossover
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform PMX between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two child chromosomes
        """
        if random.random() > self.crossover_rate or len(parent1) < 3:
            return parent1.copy(), parent2.copy()
        
        length = len(parent1)
        point1 = random.randint(0, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        child1 = self._pmx_crossover(parent1, parent2, point1, point2)
        child2 = self._pmx_crossover(parent2, parent1, point1, point2)
        
        return child1, child2
    
    def _pmx_crossover(self, parent1: Chromosome, parent2: Chromosome,
                      start: int, end: int) -> Chromosome:
        """
        Perform PMX crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            start: Start index
            end: End index
            
        Returns:
            Child chromosome
        """
        length = len(parent1)
        child_genes = [None] * length
        
        # Copy mapping section from parent1
        mapping = {}
        for i in range(start, end + 1):
            child_genes[i] = parent1[i]
            mapping[parent1[i]] = parent2[i]
        
        # Fill remaining positions
        for i in range(length):
            if i < start or i > end:
                gene = parent2[i]
                
                # Follow mapping chain
                while gene in mapping:
                    gene = mapping[gene]
                
                child_genes[i] = gene
        
        return Chromosome(child_genes, parent1.depot_id)


class CycleCrossover(CrossoverOperator):
    """
    Cycle Crossover (CX) operator.
    Preserves absolute positions of genes.
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize Cycle Crossover.
        
        Args:
            crossover_rate: Probability of performing crossover
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform Cycle Crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two child chromosomes
        """
        if random.random() > self.crossover_rate or len(parent1) < 3:
            return parent1.copy(), parent2.copy()
        
        length = len(parent1)
        child1_genes = [None] * length
        child2_genes = [None] * length
        
        # Find cycles
        visited = [False] * length
        cycles = []
        
        for i in range(length):
            if not visited[i]:
                cycle = []
                current = i
                
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    
                    # Find position of parent1[current] in parent2
                    gene_to_find = parent1[current]
                    current = parent2.genes.index(gene_to_find)
                
                cycles.append(cycle)
        
        # Create children by alternating cycles
        for cycle_idx, cycle in enumerate(cycles):
            source_parent = parent1 if cycle_idx % 2 == 0 else parent2
            
            for position in cycle:
                child1_genes[position] = source_parent[position]
                child2_genes[position] = parent2[position] if source_parent is parent1 else parent1[position]
        
        child1 = Chromosome(child1_genes, parent1.depot_id)
        child2 = Chromosome(child2_genes, parent1.depot_id)
        
        return child1, child2