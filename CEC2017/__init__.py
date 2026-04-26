"""
CEC2017 Benchmark Suite — Rao-1, Rao-2, Rao-3 & FISA.

This package provides tools for benchmarking metaheuristic optimization
algorithms on the CEC2017 test suite.
"""

__version__ = "0.1.1"
__author__ = "Lakshya Maheshwari"
__email__ = "lakshymaheshwari27@gmail.com"

from . import algorithms
from . import functions
from . import utils
from . import visualization
from .results import save_results

__all__ = ["algorithms", "functions", "utils", "visualization", "save_results"]
