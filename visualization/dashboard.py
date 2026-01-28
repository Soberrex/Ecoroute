"""
Dashboard for visualizing EcoRoute optimization results.
Provides real-time visualization of GA progress and route maps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
import seaborn as sns
from domain.route import Route
from ga.chromosome import Chromosome
from ga.genetic_algorithm import GeneticAlgorithm


class RouteVisualizer:
    """
    Visualizes routes on a 2D map.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize route visualizer.
        
        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_route(self, route: Route, title: str = "Delivery Route", 
                   show_labels: bool = True, save_path: Optional[str] = None) -> Figure:
        """
        Plot a single route.
        
        Args:
            route: Route to visualize
            title: Plot title
            show_labels: Whether to show location labels
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get route coordinates
        coords = route.get_route_coordinates()
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        # Plot route path
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6, label='Route Path')
        ax.plot(x_coords, y_coords, 'bo', markersize=6, alpha=0.8)
        
        # Highlight depot
        depot_coords = (route.depot.x, route.depot.y)
        ax.plot(depot_coords[0], depot_coords[1], 'rs', markersize=12, 
                label='Depot', markeredgecolor='black', markeredgewidth=2)
        
        # Highlight locations
        for i, location in enumerate(route.locations):
            ax.plot(location.x, location.y, 'go', markersize=8)
            
            if show_labels:
                # Add location label with demand
                label = f"L{location.id}\n{location.demand_weight}kg"
                ax.annotate(label, (location.x, location.y), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="yellow", alpha=0.7))
        
        # Add arrows to show direction
        for i in range(len(x_coords) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            ax.arrow(x_coords[i], y_coords[i], dx*0.8, dy*0.8, 
                    head_width=1, head_length=2, fc='red', ec='red', alpha=0.5)
        
        # Set plot properties
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add statistics
        distance = route.calculate_total_distance()
        total_time = route.calculate_total_time()
        total_load = route.calculate_total_load()
        
        stats_text = (f"Distance: {distance:.2f} km\n"
                     f"Total Time: {total_time:.2f} min\n"
                     f"Total Load: {total_load:.2f} kg\n"
                     f"Locations: {len(route)}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_routes(self, routes: List[Route], title: str = "Multiple Routes",
                            save_path: Optional[str] = None) -> Figure:
        """
        Plot multiple routes with different colors.
        
        Args:
            routes: List of routes to visualize
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Colors for different routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        
        # Plot each route
        for idx, route in enumerate(routes):
            coords = route.get_route_coordinates()
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            
            # Plot route
            ax.plot(x_coords, y_coords, '-', color=colors[idx], linewidth=2, 
                   alpha=0.7, label=f"Vehicle {route.vehicle.id}")
            ax.plot(x_coords, y_coords, 'o', color=colors[idx], markersize=6, alpha=0.8)
        
        # Highlight all depots (should be same for all routes)
        if routes:
            depot = routes[0].depot
            ax.plot(depot.x, depot.y, 'rs', markersize=15, 
                   label='Depot', markeredgecolor='black', markeredgewidth=2)
        
        # Add statistics
        total_distance = sum(route.calculate_total_distance() for route in routes)
        total_locations = sum(len(route) for route in routes)
        total_load = sum(route.calculate_total_load() for route in routes)
        
        stats_text = (f"Total Distance: {total_distance:.2f} km\n"
                     f"Total Locations: {total_locations}\n"
                     f"Total Load: {total_load:.2f} kg\n"
                     f"Number of Routes: {len(routes)}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_chromosome(self, chromosome: Chromosome, title: str = "Solution Routes",
                       save_path: Optional[str] = None) -> Figure:
        """
        Plot routes from a chromosome solution.
        
        Args:
            chromosome: Chromosome to visualize
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib Figure
        """
        if not chromosome.routes:
            raise ValueError("Chromosome has no decoded routes")
        
        return self.plot_multiple_routes(chromosome.routes, title, save_path)


class ConvergencePlotter:
    """
    Plots convergence metrics from GA optimization.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize convergence plotter.
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
    
    def plot_convergence(self, history: List[Dict[str, Any]], 
                        title: str = "GA Convergence Analysis",
                        save_path: Optional[str] = None) -> Figure:
        """
        Plot convergence metrics from GA history.
        
        Args:
            history: List of generation statistics
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib Figure
        """
        if not history:
            raise ValueError("No history data provided")
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        generations = list(range(len(history)))
        best_costs = [h.get('best_cost', 0) for h in history]
        avg_costs = [h.get('avg_cost', 0) for h in history]
        best_fitness = [h.get('best_fitness', 0) for h in history]
        avg_fitness = [h.get('avg_fitness', 0) for h in history]
        diversity = [h.get('population_diversity', 0) for h in history]
        feasible_counts = [h.get('feasible_count', 0) for h in history]
        
        # Plot 1: Cost convergence
        ax1 = axes[0, 0]
        ax1.plot(generations, best_costs, 'b-', linewidth=2, label='Best Cost')
        ax1.plot(generations, avg_costs, 'r--', linewidth=1.5, label='Average Cost', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Cost ($)')
        ax1.set_title('Cost Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fitness progression
        ax2 = axes[0, 1]
        ax2.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
        ax2.plot(generations, avg_fitness, 'y--', linewidth=1.5, label='Average Fitness', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness Progression')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Population diversity
        ax3 = axes[0, 2]
        ax3.plot(generations, diversity, 'm-', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity')
        ax3.set_title('Population Diversity')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feasible solutions
        ax4 = axes[1, 0]
        ax4.plot(generations, feasible_counts, 'c-', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Feasible Solutions')
        ax4.set_title('Feasible Solutions Count')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cost distribution over time
        ax5 = axes[1, 1]
        # Show cost distribution at selected generations
        sample_gens = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]
        for i, gen in enumerate(sample_gens):
            if gen < len(history):
                # Create a violin plot-like visualization
                # (Simplified - in production would use actual population data)
                cost = history[gen]['avg_cost']
                std = history[gen].get('std_cost', 0)
                ax5.errorbar(gen, cost, yerr=std, fmt='o', 
                           label=f'Gen {gen}' if i == 0 else "")
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Cost ($)')
        ax5.set_title('Cost Distribution Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Improvement rate
        ax6 = axes[1, 2]
        improvements = []
        for i in range(1, len(best_costs)):
            improvement = (best_costs[i-1] - best_costs[i]) / best_costs[i-1] * 100
            improvements.append(max(0, improvement))
        
        ax6.bar(range(1, len(best_costs)), improvements, alpha=0.7)
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('Improvement Rate per Generation')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(self, histories: Dict[str, List[Dict[str, Any]]],
                       title: str = "Algorithm Comparison",
                       save_path: Optional[str] = None) -> Figure:
        """
        Compare convergence of different algorithms.
        
        Args:
            histories: Dictionary mapping algorithm names to histories
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        for idx, (algo_name, history) in enumerate(histories.items()):
            generations = list(range(len(history)))
            best_costs = [h.get('best_cost', 0) for h in history]
            avg_costs = [h.get('avg_cost', 0) for h in history]
            diversity = [h.get('population_diversity', 0) for h in history]
            
            color = colors[idx]
            
            # Plot 1: Best cost comparison
            axes[0, 0].plot(generations, best_costs, '-', color=color, 
                          linewidth=2, label=algo_name)
            
            # Plot 2: Average cost comparison
            axes[0, 1].plot(generations, avg_costs, '--', color=color, 
                          linewidth=1.5, label=algo_name, alpha=0.7)
            
            # Plot 3: Diversity comparison
            axes[1, 0].plot(generations, diversity, '-', color=color, 
                          linewidth=1.5, label=algo_name, alpha=0.7)
            
            # Plot 4: Final cost comparison (bar chart)
            if history:
                final_cost = history[-1].get('best_cost', 0)
                axes[1, 1].bar(idx, final_cost, color=color, alpha=0.7, label=algo_name)
        
        # Set subplot properties
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Best Cost ($)')
        axes[0, 0].set_title('Best Cost Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Average Cost ($)')
        axes[0, 1].set_title('Average Cost Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Diversity')
        axes[1, 0].set_title('Diversity Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Algorithm')
        axes[1, 1].set_ylabel('Final Best Cost ($)')
        axes[1, 1].set_title('Final Solution Comparison')
        axes[1, 1].set_xticks(range(len(histories)))
        axes[1, 1].set_xticklabels(list(histories.keys()), rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class Dashboard:
    """
    Main dashboard for real-time visualization of GA optimization.
    """
    
    def __init__(self, ga_instance: GeneticAlgorithm, update_interval: int = 1):
        """
        Initialize dashboard.
        
        Args:
            ga_instance: GeneticAlgorithm instance to monitor
            update_interval: Update interval in seconds
        """
        self.ga = ga_instance
        self.update_interval = update_interval
        self.fig = None
        self.axs = None
        self.animation = None
        
        # Initialize visualization components
        self.route_visualizer = RouteVisualizer()
        self.convergence_plotter = ConvergencePlotter()
        
        # Data buffers for real-time plotting
        self.history_buffer = []
        self.best_solutions_buffer = []
    
    def create_dashboard(self) -> None:
        """Create the main dashboard figure."""
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        self.fig.suptitle('EcoRoute Optimization Dashboard', fontsize=16, fontweight='bold')
    
    def update_dashboard(self, frame: int) -> None:
        """
        Update dashboard with current GA state.
        
        Args:
            frame: Animation frame number
        """
        if not self.ga.population:
            return
        
        # Clear axes
        for ax in self.axs.flat:
            ax.clear()
        
        # Get current data
        stats = self.ga.population.get_statistics()
        self.history_buffer.append(stats)
        
        best_chrom = self.ga.population.get_best_chromosome()
        if best_chrom and best_chrom.routes:
            self.best_solutions_buffer.append(best_chrom)
        
        # Plot 1: Current best route
        ax1 = self.axs[0, 0]
        if best_chrom and best_chrom.routes and len(best_chrom.routes) > 0:
            # Plot first route of best solution
            route = best_chrom.routes[0]
            coords = route.get_route_coordinates()
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            
            ax1.plot(x_coords, y_coords, 'b-', linewidth=2)
            ax1.plot(x_coords, y_coords, 'bo', markersize=4)
            ax1.plot(route.depot.x, route.depot.y, 'rs', markersize=10)
            
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.set_title(f'Best Route (Cost: ${best_chrom.get_total_cost():.2f})')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
        
        # Plot 2: Cost convergence
        ax2 = self.axs[0, 1]
        if len(self.history_buffer) > 1:
            generations = list(range(len(self.history_buffer)))
            best_costs = [h.get('best_cost', 0) for h in self.history_buffer]
            avg_costs = [h.get('avg_cost', 0) for h in self.history_buffer]
            
            ax2.plot(generations, best_costs, 'b-', linewidth=2, label='Best Cost')
            ax2.plot(generations, avg_costs, 'r--', linewidth=1.5, label='Avg Cost')
            
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Cost ($)')
            ax2.set_title('Cost Convergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Diversity and feasibility
        ax3 = self.axs[1, 0]
        if len(self.history_buffer) > 1:
            generations = list(range(len(self.history_buffer)))
            diversity = [h.get('population_diversity', 0) for h in self.history_buffer]
            feasible = [h.get('feasible_count', 0) for h in self.history_buffer]
            
            ax3.plot(generations, diversity, 'g-', linewidth=2, label='Diversity')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Diversity', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
            ax3_secondary = ax3.twinx()
            ax3_secondary.plot(generations, feasible, 'm--', linewidth=1.5, label='Feasible')
            ax3_secondary.set_ylabel('Feasible Count', color='m')
            ax3_secondary.tick_params(axis='y', labelcolor='m')
            
            ax3.set_title('Diversity and Feasibility')
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_secondary.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 4: Current statistics
        ax4 = self.axs[1, 1]
        ax4.axis('off')
        
        if stats:
            stats_text = (
                f"Generation: {stats.get('generation', 0)}\n"
                f"Best Cost: ${stats.get('best_cost', 0):.2f}\n"
                f"Average Cost: ${stats.get('avg_cost', 0):.2f}\n"
                f"Best Fitness: {stats.get('best_fitness', 0):.6f}\n"
                f"Diversity: {stats.get('population_diversity', 0):.3f}\n"
                f"Feasible Solutions: {stats.get('feasible_count', 0)}\n"
                f"Total Locations: {len([loc for loc in self.ga.locations.values() if not loc.is_depot])}\n"
                f"Time Elapsed: {stats.get('elapsed_seconds', 0):.1f}s"
            )
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def start_real_time_visualization(self) -> None:
        """Start real-time visualization animation."""
        if self.fig is None:
            self.create_dashboard()
        
        self.animation = FuncAnimation(
            self.fig, self.update_dashboard,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            cache_frame_data=False
        )
        
        plt.show()
    
    def stop_visualization(self) -> None:
        """Stop the visualization animation."""
        if self.animation:
            self.animation.event_source.stop()
    
    def save_snapshot(self, filename: str = None) -> None:
        """
        Save current dashboard state.
        
        Args:
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_snapshot_{timestamp}.png"
        
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dashboard snapshot saved as {filename}")
    
    def generate_final_report(self, best_solution: Chromosome, 
                            output_dir: str = "reports") -> None:
        """
        Generate final optimization report with visualizations.
        
        Args:
            best_solution: Best solution found
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Plot convergence
        conv_fig = self.convergence_plotter.plot_convergence(
            self.ga.get_history(),
            title=f"EcoRoute Optimization Convergence ({timestamp})"
        )
        conv_fig.savefig(f"{output_dir}/convergence_{timestamp}.png", 
                        dpi=300, bbox_inches='tight')
        
        # 2. Plot best solution routes
        if best_solution and best_solution.routes:
            route_fig = self.route_visualizer.plot_chromosome(
                best_solution,
                title=f"Best Solution Routes ({timestamp})"
            )
            route_fig.savefig(f"{output_dir}/best_routes_{timestamp}.png",
                            dpi=300, bbox_inches='tight')
        
        # 3. Create summary text file
        summary_file = f"{output_dir}/summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ECOROUTE OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total generations: {len(self.ga.get_history())}\n")
            
            if self.ga.get_history():
                first = self.ga.get_history()[0]
                last = self.ga.get_history()[-1]
                
                improvement = first['best_cost'] - last['best_cost']
                improvement_pct = (improvement / first['best_cost']) * 100
                
                f.write(f"\nCOST IMPROVEMENT:\n")
                f.write(f"  Initial best cost: ${first['best_cost']:.2f}\n")
                f.write(f"  Final best cost:   ${last['best_cost']:.2f}\n")
                f.write(f"  Improvement:       ${improvement:.2f} ({improvement_pct:.1f}%)\n")
            
            if best_solution:
                f.write(f"\nBEST SOLUTION:\n")
                f.write(f"  Total cost: ${best_solution.get_total_cost():.2f}\n")
                f.write(f"  Fitness: {best_solution.fitness:.6f}\n")
                
                if best_solution.cost_evaluation:
                    eval_data = best_solution.cost_evaluation
                    f.write(f"  Number of routes: {eval_data.get('num_routes', 0)}\n")
                    f.write(f"  Total locations: {eval_data.get('total_locations', 0)}\n")
                    f.write(f"  Total distance: {eval_data.get('distance_km', 0):.2f} km\n")
        
        print(f"Final report generated in {output_dir}/")