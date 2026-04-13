"""
CEC2017-Rao2: CEC2017 benchmark suite using Rao-2 optimization algorithm.

This package provides tools for benchmarking optimization algorithms
on the CEC2017 test suite using the Rao-2 algorithm.
"""

__version__ = "0.1.0"
__author__ = "Lakshya Maheshwari"
__email__ = "your.email@example.com"  # Replace with your actual email

from . import algorithms
from . import functions
from . import utils
from . import visualization
from .results import save_results

__all__ = ["algorithms", "functions", "utils", "visualization", "save_results"]