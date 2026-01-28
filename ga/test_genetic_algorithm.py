#!/usr/bin/env python3
"""
Test script for EcoRoute Genetic Algorithm components.
"""

import random
import json
import tempfile
from domain.location import Location
from domain.vehicle import Vehicle
from cost.cost_engine import CostEngine
from ga.chromosome import Chromosome
from ga.population import Population
from ga.selection import TournamentSelection, RouletteWheelSelection
from ga.crossover import OrderedCrossover
from ga.mutation import SwapMutation, InversionMutation
from ga.genetic_algorithm import GeneticAlgorithm


def test_ga_components():
    """Test basic functionality of GA components."""
    
    print("Testing EcoRoute Genetic Algorithm Components...")
    print("=" * 60)
    
    # Create test data
    random.seed(42)  # For reproducible tests
    
    # Create depot and locations
    depot = Location(id=0, x=0, y=0, is_depot=True)
    
    locations = {}
    locations[0] = depot
    
    for i in range(1, 11):
        locations[i] = Location(
            id=i,
            x=random.uniform(0, 100),
            y=random.uniform(0, 100),
            demand_weight=random.uniform(10, 50),
            name=f"Customer {i}"
        )
    
    print(f"Created {len(locations)-1} customer locations")
    
    # Create vehicles
    vehicles = [
        Vehicle(max_capacity=200, fuel_efficiency=0.2, max_route_time=480),
        Vehicle(max_capacity=150, fuel_efficiency=0.15, max_route_time=360),
    ]
    
    print(f"Created {len(vehicles)} vehicles")
    
    # Create cost engine
    cost_engine = CostEngine()
    
    # Test Chromosome
    print("\n" + "=" * 60)
    print("Testing Chromosome Class...")
    
    # Create random chromosome
    chrom = Chromosome.create_random(locations)
    print(f"Random chromosome: {chrom}")
    print(f"Length: {len(chrom)} genes")
    
    # Test chromosome validation
    is_valid = chrom.is_valid(locations)
    print(f"Is valid: {is_valid}")
    
    # Test chromosome decoding and evaluation
    chrom.decode(locations, vehicles, cost_engine)
    print(f"Decoded chromosome cost: ${chrom.get_total_cost():.2f}")
    print(f"Chromosome fitness: {chrom.fitness:.6f}")
    
    # Test chromosome copy
    chrom_copy = chrom.copy()
    print(f"Copy created. Original fitness: {chrom.fitness}, Copy fitness: {chrom_copy.fitness}")
    
    # Test Population
    print("\n" + "=" * 60)
    print("Testing Population Class...")
    
    population = Population(size=20)
    population.initialize_mixed(locations, depot_id=0, greedy_ratio=0.2)
    population.evaluate(locations, vehicles, cost_engine)
    
    print(f"Population size: {len(population)}")
    print(f"Best chromosome cost: ${population.best_chromosome.get_total_cost():.2f}")
    
    stats = population.get_statistics()
    print(f"Population statistics:")
    print(f"  Best fitness: {stats['best_fitness']:.6f}")
    print(f"  Average fitness: {stats['avg_fitness']:.6f}")
    print(f"  Diversity: {stats['population_diversity']:.3f}")
    
    # Test elite selection
    elite = population.select_elite(elite_ratio=0.1)
    print(f"Selected {len(elite)} elite chromosomes")
    
    # Test Selection Operators
    print("\n" + "=" * 60)
    print("Testing Selection Operators...")
    
    # Tournament selection
    tournament_selector = TournamentSelection(tournament_size=3)
    selected1 = tournament_selector.select(population.chromosomes)
    print(f"Tournament selection: {selected1}")
    
    # Roulette wheel selection
    roulette_selector = RouletteWheelSelection(scaling='linear')
    selected2 = roulette_selector.select(population.chromosomes)
    print(f"Roulette selection: {selected2}")
    
    # Test Crossover Operators
    print("\n" + "=" * 60)
    print("Testing Crossover Operators...")
    
    parent1 = Chromosome.create_random(locations)
    parent2 = Chromosome.create_random(locations)
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    crossover = OrderedCrossover(crossover_rate=1.0)  # Always crossover
    child1, child2 = crossover.crossover(parent1, parent2)
    
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")
    
    # Verify children are valid
    print(f"Child 1 valid: {child1.is_valid(locations)}")
    print(f"Child 2 valid: {child2.is_valid(locations)}")
    
    # Test Mutation Operators
    print("\n" + "=" * 60)
    print("Testing Mutation Operators...")
    
    # Swap mutation
    swap_mutator = SwapMutation(mutation_rate=1.0)  # Always mutate
    mutated1 = swap_mutator.mutate(child1)
    print(f"Original: {child1}")
    print(f"Mutated (swap): {mutated1}")
    
    # Inversion mutation
    inversion_mutator = InversionMutation(mutation_rate=1.0)
    mutated2 = inversion_mutator.mutate(child2)
    print(f"Original: {child2}")
    print(f"Mutated (inversion): {mutated2}")
    
    # Test full Genetic Algorithm
    print("\n" + "=" * 60)
    print("Testing Full Genetic Algorithm...")
    
    # Create GA configuration
    config = {
        'population_size': 30,
        'generations': 10,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism_rate': 0.1,
        'tournament_size': 3,
        'adaptive_mutation': True,
        'use_2opt': True,
        '2opt_max_iterations': 5
    }
    
    # Create and run GA
    ga = GeneticAlgorithm(cost_engine, vehicles, locations, config)
    
    # Run for a few generations
    best_solution = ga.run(generations=5, population_size=30, early_stopping=False)
    
    print(f"\nBest solution found: {best_solution}")
    print(f"Best cost: ${best_solution.get_total_cost():.2f}")
    
    # Print GA history
    history = ga.get_history()
    print(f"\nEvolution history ({len(history)} generations):")
    for i, gen in enumerate(history[:3]):  # Show first 3 generations
        print(f"  Gen {i}: Best=${gen['best_cost']:.2f}, Avg=${gen['avg_cost']:.2f}")
    
    if len(history) > 3:
        print(f"  ... and {len(history) - 3} more generations")
    
    print("\nGA components test completed successfully!")


if __name__ == "__main__":
    test_ga_components()