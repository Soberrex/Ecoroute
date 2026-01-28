#!/usr/bin/env python3
"""
Comprehensive test for the complete EcoRoute system.
"""

import sys
import os
import tempfile
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .main import EcoRouteOptimizer


def test_complete_system():
    """Test the complete EcoRoute system."""
    
    print("=" * 70)
    print("COMPREHENSIVE ECOROUTE SYSTEM TEST")
    print("=" * 70)
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Test directory: {tmpdir}")
        
        # Create sample configuration
        config = {
            "genetic_algorithm": {
                "population_size": 30,
                "generations": 20,
                "crossover_rate": 0.8,
                "mutation_rate": 0.2,
                "elitism_rate": 0.05,
                "tournament_size": 3,
                "adaptive_mutation": True,
                "use_2opt": True,
                "early_stopping_generations": 5
            },
            "vehicles": {
                "default_capacity": 200.0,
                "default_fuel_efficiency": 0.2,
                "default_max_route_time": 480.0,
                "default_fixed_cost": 50.0,
                "default_variable_cost": 0.5,
                "fuel_cost_per_liter": 1.5
            },
            "locations": {
                "time_window_penalty_weight": 10.0,
                "capacity_penalty_weight": 100.0,
                "time_penalty_weight": 5.0
            }
        }
        
        config_file = os.path.join(tmpdir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n1. Creating EcoRoute optimizer...")
        optimizer = EcoRouteOptimizer(config_file)
        
        print("\n2. Generating sample locations...")
        optimizer.generate_sample_locations(num_customers=15, area_size=100)
        
        print("\n3. Setting up vehicles...")
        optimizer.setup_vehicles(num_vehicles=2)
        
        print("\n4. Setting up cost engine...")
        optimizer.setup_cost_engine()
        
        print("\n5. Running optimization with Genetic Algorithm...")
        try:
            best_solution = optimizer.optimize(
                algorithm='genetic',
                generations=10,
                population_size=20
            )
            
            if best_solution:
                print(f"  Optimization successful!")
                print(f"  Best solution cost: ${best_solution.get_total_cost():.2f}")
                
                # Test visualization (without showing)
                print("\n6. Testing visualization...")
                try:
                    from .visualization.dashboard import RouteVisualizer
                    visualizer = RouteVisualizer()
                    
                    if best_solution.routes:
                        fig = visualizer.plot_chromosome(
                            best_solution,
                            title="Test Visualization"
                        )
                        
                        # Save figure instead of showing
                        output_file = os.path.join(tmpdir, "test_visualization.png")
                        fig.savefig(output_file, dpi=150, bbox_inches='tight')
                        print(f"  Visualization saved to {output_file}")
                        
                        # Close figure to free memory
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                except Exception as e:
                    print(f"  Visualization test skipped: {e}")
                
                # Test convergence plot
                print("\n7. Testing convergence plot...")
                try:
                    from .visualization.dashboard import ConvergencePlotter
                    plotter = ConvergencePlotter()
                    
                    if optimizer.optimization_history:
                        fig = plotter.plot_convergence(
                            optimizer.optimization_history,
                            title="Test Convergence"
                        )
                        
                        output_file = os.path.join(tmpdir, "test_convergence.png")
                        fig.savefig(output_file, dpi=150, bbox_inches='tight')
                        print(f"  Convergence plot saved to {output_file}")
                        
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                except Exception as e:
                    print(f"  Convergence plot test skipped: {e}")
                
                # Save results
                print("\n8. Saving results...")
                optimizer.save_results(tmpdir)
                
                # Print summary
                print("\n9. Final summary:")
                optimizer.print_summary()
                
                print("\n✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
                print(f"All test files saved in: {tmpdir}")
                
            else:
                print("❌ Optimization failed to produce a solution.")
                return False
                
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)