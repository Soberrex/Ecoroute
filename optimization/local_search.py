"""
Local search optimization techniques for route improvement.
Implements 2-opt, 3-opt, and other local search methods.
"""

from typing import List, Optional, Tuple
import random
import numpy as np
from domain.route import Route
from ga.chromosome import Chromosome
from cost.cost_engine import CostEngine


class TwoOptOptimizer:
    """
    2-opt local search optimizer.
    Improves routes by reversing segments to eliminate crossing paths.
    
    Why it improves convergence:
    - Directly targets route distance by eliminating route crossings
    - Simple and effective for Euclidean TSP-like problems
    - Can be applied as a post-processing step or during GA
    """
    
    def __init__(self, max_iterations: int = 100, improve_threshold: float = 0.001):
        """
        Initialize 2-opt optimizer.
        
        Args:
            max_iterations: Maximum iterations without improvement
            improve_threshold: Minimum improvement ratio to continue
        """
        self.max_iterations = max_iterations
        self.improve_threshold = improve_threshold
    
    def optimize(self, chromosome: Chromosome, locations: dict, vehicles: list,
                 cost_engine: CostEngine) -> Chromosome:
        """
        Apply 2-opt optimization to a chromosome.
        
        Args:
            chromosome: Chromosome to optimize
            locations: Dictionary of locations
            vehicles: List of vehicles
            cost_engine: Cost evaluation engine
            
        Returns:
            Optimized chromosome
        """
        # Ensure chromosome is evaluated
        if not chromosome.is_decoded:
            chromosome.decode(locations, vehicles, cost_engine)
        
        best_chrom = chromosome.copy()
        best_cost = best_chrom.get_total_cost()
        
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            improved = False
            length = len(best_chrom)
            
            # Try all possible 2-opt swaps
            for i in range(1, length - 1):
                for j in range(i + 1, length):
                    # Create new route by reversing segment i..j
                    new_genes = best_chrom.genes[:]
                    new_genes[i:j+1] = reversed(new_genes[i:j+1])
                    
                    # Create and evaluate new chromosome
                    new_chrom = Chromosome(new_genes, best_chrom.depot_id)
                    new_chrom.decode(locations, vehicles, cost_engine)
                    new_cost = new_chrom.get_total_cost()
                    
                    # Keep if better
                    if new_cost < best_cost * (1 - self.improve_threshold):
                        best_chrom = new_chrom
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            iterations += 1
        
        return best_chrom
    
    def optimize_route(self, route: Route) -> Route:
        """
        Apply 2-opt to a single route.
        
        Args:
            route: Route to optimize
            
        Returns:
            Optimized route
        """
        # Extract location IDs
        location_ids = [loc.id for loc in route.locations]
        
        improved = True
        best_distance = route.calculate_total_distance()
        best_route = route
        
        while improved:
            improved = False
            length = len(location_ids)
            
            for i in range(length - 1):
                for j in range(i + 2, length):
                    # Try reversing segment i+1..j
                    new_route_ids = location_ids[:]
                    new_route_ids[i+1:j+1] = reversed(new_route_ids[i+1:j+1])
                    
                    # Create new route
                    new_locations = [route.locations[idx] for idx in 
                                    [location_ids.index(id) for id in new_route_ids]]
                    new_route = Route(
                        vehicle=route.vehicle,
                        depot=route.depot,
                        locations=new_locations,
                        speed_kmh=route.speed_kmh
                    )
                    
                    new_distance = new_route.calculate_total_distance()
                    
                    if new_distance < best_distance:
                        location_ids = new_route_ids
                        best_distance = new_distance
                        best_route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route


class ThreeOptOptimizer:
    """
    3-opt local search optimizer.
    More powerful than 2-opt but computationally more expensive.
    
    Why it improves convergence:
    - Can make more complex route changes than 2-opt
    - Better at escaping local optima
    - Particularly effective for larger problem instances
    """
    
    def __init__(self, max_iterations: int = 50):
        """
        Initialize 3-opt optimizer.
        
        Args:
            max_iterations: Maximum iterations
        """
        self.max_iterations = max_iterations
    
    def optimize_route(self, route: Route) -> Route:
        """
        Apply 3-opt optimization to a route.
        
        Args:
            route: Route to optimize
            
        Returns:
            Optimized route
        """
        location_ids = [loc.id for loc in route.locations]
        best_distance = route.calculate_total_distance()
        best_route = route
        
        for _ in range(self.max_iterations):
            improved = False
            length = len(location_ids)
            
            for i in range(length):
                for j in range(i + 2, length):
                    for k in range(j + 2, length):
                        # Try different 3-opt moves
                        moves = self._three_opt_moves(location_ids, i, j, k)
                        
                        for move in moves:
                            # Create and evaluate new route
                            new_locations = [route.locations[idx] for idx in 
                                            [location_ids.index(id) for id in move]]
                            new_route = Route(
                                vehicle=route.vehicle,
                                depot=route.depot,
                                locations=new_locations,
                                speed_kmh=route.speed_kmh
                            )
                            
                            new_distance = new_route.calculate_total_distance()
                            
                            if new_distance < best_distance:
                                location_ids = move
                                best_distance = new_distance
                                best_route = new_route
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_route
    
    def _three_opt_moves(self, route: List[int], i: int, j: int, k: int) -> List[List[int]]:
        """
        Generate all possible 3-opt moves.
        
        Args:
            route: Current route
            i, j, k: Cut points
            
        Returns:
            List of possible new routes
        """
        # Note: 3-opt has 7 possible reconnections
        # For simplicity, implement the most common ones
        moves = []
        
        # Standard 3-opt moves
        # 1. Reverse segment (j, k)
        move1 = route[:i+1] + route[i+1:j+1] + list(reversed(route[j+1:k+1])) + route[k+1:]
        moves.append(move1)
        
        # 2. Reverse segment (i, j)
        move2 = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:k+1] + route[k+1:]
        moves.append(move2)
        
        # 3. Swap segments
        move3 = route[:i+1] + route[j+1:k+1] + route[i+1:j+1] + route[k+1:]
        moves.append(move3)
        
        return moves


class LocalSearch:
    """
    Comprehensive local search framework.
    Combines multiple local search techniques.
    """
    
    def __init__(self, use_2opt: bool = True, use_3opt: bool = False,
                 intensity: str = 'medium'):
        """
        Initialize local search.
        
        Args:
            use_2opt: Whether to use 2-opt
            use_3opt: Whether to use 3-opt
            intensity: Search intensity ('light', 'medium', 'aggressive')
        """
        self.use_2opt = use_2opt
        self.use_3opt = use_3opt
        
        # Configure based on intensity
        if intensity == 'light':
            self.two_opt_iterations = 20
            self.three_opt_iterations = 10
        elif intensity == 'aggressive':
            self.two_opt_iterations = 100
            self.three_opt_iterations = 50
        else:  # medium
            self.two_opt_iterations = 50
            self.three_opt_iterations = 20
        
        if use_2opt:
            self.two_opt = TwoOptOptimizer(max_iterations=self.two_opt_iterations)
        if use_3opt:
            self.three_opt = ThreeOptOptimizer(max_iterations=self.three_opt_iterations)
    
    def optimize_chromosome(self, chromosome: Chromosome, locations: dict, 
                           vehicles: list, cost_engine: CostEngine) -> Chromosome:
        """
        Apply local search to a chromosome.
        
        Args:
            chromosome: Chromosome to optimize
            locations: Dictionary of locations
            vehicles: List of vehicles
            cost_engine: Cost evaluation engine
            
        Returns:
            Optimized chromosome
        """
        # Ensure chromosome is evaluated
        if not chromosome.is_decoded:
            chromosome.decode(locations, vehicles, cost_engine)
        
        optimized = chromosome.copy()
        
        # Apply 2-opt
        if self.use_2opt:
            optimized = self.two_opt.optimize(optimized, locations, vehicles, cost_engine)
        
        # Apply 3-opt (more expensive, use sparingly)
        if self.use_3opt and len(optimized) > 10:
            # For 3-opt, we need to optimize each route separately
            if optimized.routes:
                optimized_routes = []
                for route in optimized.routes:
                    optimized_route = self.three_opt.optimize_route(route)
                    optimized_routes.append(optimized_route)
                
                # Recreate chromosome from optimized routes
                # (This is simplified - in production, would need proper reconstruction)
                pass
        
        return optimized
    
    def optimize_population(
        self,
        population: List[Chromosome],
        locations: dict,
        vehicles: list,
        cost_engine: CostEngine,
        elite_ratio: float = 0.1
    ) -> List[Chromosome]:
        """
        Apply local search to elite members of a population.
        
        Args:
            population: List of chromosomes
            elite_ratio: Proportion to apply local search to
            locations: Dictionary of locations
            vehicles: List of vehicles
            cost_engine: Cost evaluation engine
            
        Returns:
            Optimized population
        """
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Apply local search to elite
        num_elite = max(1, int(len(population) * elite_ratio))
        
        for i in range(num_elite):
            optimized = self.optimize_chromosome(
                sorted_pop[i], locations, vehicles, cost_engine
            )
            sorted_pop[i] = optimized
        
        return sorted_pop