"""
Chromosome representation for Genetic Algorithm.
Encodes delivery routes as permutations of location IDs.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from domain.route import Route
from domain.location import Location
from domain.vehicle import Vehicle
from cost.cost_engine import CostEngine


class Chromosome:
    """
    Represents a potential solution (route sequence) in the Genetic Algorithm.
    
    Attributes:
        genes: List of location IDs representing the delivery sequence
        depot_id: ID of the depot location
        fitness: Fitness value of this chromosome
        routes: List of Route objects after decoding
        cost_evaluation: Detailed cost evaluation
        is_decoded: Whether the chromosome has been decoded to routes
    """
    
    def __init__(self, genes: List[int], depot_id: int = 0):
        """
        Initialize a chromosome with gene sequence.
        
        Args:
            genes: Sequence of location IDs (excluding depot)
            depot_id: ID of the depot location
        """
        self.genes = genes.copy()  # Copy to avoid mutation side effects
        self.depot_id = depot_id
        self.fitness = 0.0
        self.routes: Optional[List[Route]] = None
        self.cost_evaluation: Optional[Dict[str, Any]] = None
        self.is_decoded = False
        self.total_cost = float('inf')
    
    def decode(self, locations: Dict[int, Location], vehicles: List[Vehicle], 
               cost_engine: CostEngine, max_locations_per_route: Optional[int] = None) -> None:
        """
        Decode chromosome into routes by assigning locations to vehicles.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            vehicles: List of available vehicles
            cost_engine: Cost engine for route evaluation
            max_locations_per_route: Maximum locations per route (None for unlimited)
        """
        if self.is_decoded:
            return
        
        self.routes = []
        depot = locations[self.depot_id]
        
        # Create a copy of vehicles to avoid modifying originals
        vehicle_copies = []
        for vehicle in vehicles:
            # Create a new vehicle instance with same properties
            new_vehicle = Vehicle(
                max_capacity=vehicle.max_capacity,
                fuel_efficiency=vehicle.fuel_efficiency,
                max_route_time=vehicle.max_route_time,
                fixed_cost=vehicle.fixed_cost,
                variable_cost=vehicle.variable_cost,
                fuel_cost_per_liter=vehicle.fuel_cost_per_liter
            )
            new_vehicle.current_location = depot
            vehicle_copies.append(new_vehicle)
        
        # Simple decoding: assign locations to vehicles in order
        current_vehicle_index = 0
        current_route_locations = []
        
        for gene in self.genes:
            location = locations[gene]
            
            # Check if we need to start a new route
            if max_locations_per_route and len(current_route_locations) >= max_locations_per_route:
                # Create route with current vehicle
                if current_route_locations:
                    route = Route(
                        vehicle=vehicle_copies[current_vehicle_index],
                        depot=depot,
                        locations=current_route_locations.copy()
                    )
                    self.routes.append(route)
                
                # Move to next vehicle
                current_vehicle_index = (current_vehicle_index + 1) % len(vehicle_copies)
                current_route_locations = []
            
            # Add location to current route
            current_route_locations.append(location)
        
        # Add the last route
        if current_route_locations:
            if current_vehicle_index >= len(vehicle_copies):
                current_vehicle_index = 0  # Wrap around if needed
            route = Route(
                vehicle=vehicle_copies[current_vehicle_index],
                depot=depot,
                locations=current_route_locations
            )
            self.routes.append(route)
        
        # Evaluate total cost
        self._evaluate(cost_engine)
        self.is_decoded = True
    
    def _evaluate(self, cost_engine: CostEngine) -> None:
        """Evaluate the chromosome using the cost engine."""
        if not self.routes:
            self.total_cost = float('inf')
            self.fitness = 0.0
            return
        
        total_cost = 0.0
        evaluations = []
        
        for route in self.routes:
            evaluation = cost_engine.evaluate_route(route, start_time=480)
            total_cost += evaluation['total_cost']
            evaluations.append(evaluation)
        
        self.total_cost = total_cost
        self.fitness = 1.0 / (total_cost + 1e-6)  # Higher fitness is better
        self.cost_evaluation = {
            'total_cost': total_cost,
            'route_evaluations': evaluations,
            'num_routes': len(self.routes),
            'total_locations': len(self.genes)
        }
    
    def evaluate_fitness(self, locations: Dict[int, Location], vehicles: List[Vehicle],
                        cost_engine: CostEngine) -> float:
        """
        Evaluate and return fitness of this chromosome.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            vehicles: List of available vehicles
            cost_engine: Cost engine for route evaluation
            
        Returns:
            Fitness value (higher is better)
        """
        if not self.is_decoded:
            self.decode(locations, vehicles, cost_engine)
        
        return self.fitness
    
    def get_total_cost(self) -> float:
        """Get the total cost of this chromosome."""
        return self.total_cost
    
    def is_valid(self, locations: Dict[int, Location]) -> bool:
        """
        Check if chromosome is valid (contains all required locations once).
        
        Args:
            locations: Dictionary of all locations
            
        Returns:
            True if chromosome is valid
        """
        # Get all non-depot location IDs
        required_locations = {loc_id for loc_id, loc in locations.items() if not loc.is_depot}
        
        # Check if chromosome contains all required locations exactly once
        chromosome_set = set(self.genes)
        
        if chromosome_set != required_locations:
            return False
        
        # Check for duplicates
        if len(self.genes) != len(chromosome_set):
            return False
        
        return True
    
    def repair(self, locations: Dict[int, Location]) -> None:
        """
        Repair invalid chromosome by fixing missing or duplicate genes.
        
        Args:
            locations: Dictionary of all locations
        """
        # Get all non-depot location IDs
        required_locations = {loc_id for loc_id, loc in locations.items() if not loc.is_depot}
        
        # Count occurrences of each gene
        gene_count = {}
        for gene in self.genes:
            gene_count[gene] = gene_count.get(gene, 0) + 1
        
        # Remove duplicates
        unique_genes = []
        seen = set()
        for gene in self.genes:
            if gene not in seen:
                unique_genes.append(gene)
                seen.add(gene)
        
        # Add missing genes
        missing = required_locations - set(unique_genes)
        unique_genes.extend(list(missing))
        
        # Remove genes that shouldn't be there (depot)
        unique_genes = [gene for gene in unique_genes if gene in required_locations]
        
        self.genes = unique_genes
        self.is_decoded = False  # Need to re-decode after repair
    
    def copy(self) -> 'Chromosome':
        """Create a deep copy of this chromosome."""
        chrom_copy = Chromosome(self.genes, self.depot_id)
        chrom_copy.fitness = self.fitness
        chrom_copy.total_cost = self.total_cost
        chrom_copy.is_decoded = self.is_decoded
        
        if self.routes:
            # Note: Routes are not deep copied for performance
            chrom_copy.routes = self.routes
        
        if self.cost_evaluation:
            chrom_copy.cost_evaluation = self.cost_evaluation.copy()
        
        return chrom_copy
    
    def __len__(self) -> int:
        """Number of genes in the chromosome."""
        return len(self.genes)
    
    def __getitem__(self, index: int) -> int:
        """Get gene at specific index."""
        return self.genes[index]
    
    def __setitem__(self, index: int, value: int) -> None:
        """Set gene at specific index."""
        self.genes[index] = value
        self.is_decoded = False  # Invalidate decoded routes
    
    def __str__(self) -> str:
        """String representation of the chromosome."""
        genes_str = '->'.join(str(gene) for gene in self.genes)
        return f"Chromosome[{genes_str}] | Fitness: {self.fitness:.6f} | Cost: ${self.total_cost:.2f}"
    
    def __repr__(self) -> str:
        return f"Chromosome(genes={self.genes}, fitness={self.fitness})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on gene sequence."""
        if not isinstance(other, Chromosome):
            return False
        return self.genes == other.genes
    
    def __hash__(self) -> int:
        """Hash based on gene sequence."""
        return hash(tuple(self.genes))
    
    @classmethod
    def create_random(cls, locations: Dict[int, Location], depot_id: int = 0) -> 'Chromosome':
        """
        Create a random chromosome visiting all locations.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            depot_id: ID of the depot location
            
        Returns:
            Randomly generated chromosome
        """
        # Get all non-depot location IDs
        non_depot_ids = [loc_id for loc_id, loc in locations.items() if not loc.is_depot]
        
        # Create random permutation
        random.shuffle(non_depot_ids)
        
        return cls(non_depot_ids, depot_id)
    
    @classmethod
    def create_greedy(cls, locations: Dict[int, Location], depot_id: int = 0) -> 'Chromosome':
        """
        Create a chromosome using nearest-neighbor greedy algorithm.
        
        Args:
            locations: Dictionary mapping location IDs to Location objects
            depot_id: ID of the depot location
            
        Returns:
            Chromosome generated by greedy algorithm
        """
        # Get all non-depot location IDs
        non_depot_ids = [loc_id for loc_id, loc in locations.items() if not loc.is_depot]
        
        if not non_depot_ids:
            return cls([], depot_id)
        
        depot = locations[depot_id]
        unvisited = non_depot_ids.copy()
        route = []
        
        # Start from a random location
        current_id = random.choice(unvisited)
        route.append(current_id)
        unvisited.remove(current_id)
        
        while unvisited:
            current_loc = locations[current_id]
            
            # Find nearest unvisited location
            nearest_id = min(unvisited, 
                           key=lambda loc_id: current_loc.distance_to(locations[loc_id]))
            
            route.append(nearest_id)
            unvisited.remove(nearest_id)
            current_id = nearest_id
        
        return cls(route, depot_id)