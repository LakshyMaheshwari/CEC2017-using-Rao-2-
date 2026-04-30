"""
Optimization algorithms module.

Contains implementations of optimization algorithms:
Rao-1, Rao-2, Rao-3, and FISA.
"""

from .rao1 import rao1
from .rao2 import rao2
from .rao3 import rao3
from .fisa import fisa

ALGORITHMS = {"rao1": rao1, "rao2": rao2, "rao3": rao3, "fisa": fisa}

__all__ = ["rao1", "rao2", "rao3", "fisa", "ALGORITHMS"]
