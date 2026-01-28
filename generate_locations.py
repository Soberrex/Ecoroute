#!/usr/bin/env python3
"""
Generate a sample locations CSV file for EcoRoute.
"""

import csv
import random
from pathlib import Path

def generate_locations_csv(filename: str, num_customers: int = 50, area_size: float = 100.0):
    """
    Generate a sample locations CSV file.
    
    Args:
        filename: Output CSV file path
        num_customers: Number of customer locations
        area_size: Size of the area (square)
    """
    # Create data directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'id', 'x', 'y', 'demand_weight', 'service_time',
            'name', 'time_window_start', 'time_window_end', 'is_depot'
        ])
        
        # Write depot
        writer.writerow([
            0, area_size / 2, area_size / 2,  # Center coordinates
            0, 0,  # No demand, no service time
            'Central Depot',
            0, 1440,  # Open all day
            'true'
        ])
        
        # Write customers
        for i in range(1, num_customers + 1):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            demand = random.uniform(10, 50)
            service_time = random.uniform(3, 15)
            
            # Generate time window (8am to 6pm with some variation)
            start_hour = random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16])
            start_minute = random.choice([0, 15, 30, 45])
            window_start = start_hour * 60 + start_minute
            
            window_duration = random.uniform(60, 180)  # 1-3 hour windows
            window_end = window_start + window_duration
            
            writer.writerow([
                i, x, y,
                round(demand, 1),
                round(service_time, 1),
                f'Customer {i}',
                int(window_start),
                int(window_end),
                'false'
            ])
    
    print(f"Generated {num_customers} customer locations in {filename}")

if __name__ == "__main__":
    # Generate a sample dataset
    generate_locations_csv(
        filename="ecoroute/data/locations.csv",
        num_customers=50,
        area_size=100.0
    )