"""
Simulation module for EcoRoute.
Provides traffic simulation and benchmarking capabilities.
"""

from .traffic import TrafficSimulator, PeakHourSimulator, DynamicTraffic
from .benchmark import BenchmarkSuite, RandomRouter, GreedyRouter

__all__ = [
    'TrafficSimulator',
    'PeakHourSimulator',
    'DynamicTraffic',
    'BenchmarkSuite',
    'RandomRouter',
    'GreedyRouter'
]