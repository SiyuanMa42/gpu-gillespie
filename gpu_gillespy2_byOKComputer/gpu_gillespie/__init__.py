"""
GPU-Gillespie: High-performance stochastic simulation with GPU acceleration

A Python package for accelerating Gillespie stochastic simulation algorithms
using GPU parallel computing. Built on top of Gillespy2 with Numba CUDA support.
"""

__version__ = "1.0.0"
__author__ = "GPU-Gillespie Development Team"
__email__ = "contact@gpu-gillespie.org"

from .core.gillespie_gpu import GPUGillespieSimulator
from .core.parallel_simulator import ParallelSimulator
from .models.basic_models import (
    DimerizationModel, 
    EnzymeKineticsModel, 
    GeneExpressionModel
)
from .utils.performance_metrics import PerformanceMonitor
from .utils.data_processing import SimulationAnalyzer

__all__ = [
    'GPUGillespieSimulator',
    'ParallelSimulator', 
    'DimerizationModel',
    'EnzymeKineticsModel',
    'GeneExpressionModel',
    'PerformanceMonitor',
    'SimulationAnalyzer'
]