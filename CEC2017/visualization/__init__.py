"""
Visualization module.

Contains plotting functions for convergence plots, 2D contours, 3D surfaces, etc.
"""

from .plot_convergence import plot_convergence
from .plot_2d_contour import plot_2d_contour
from .plot_3d_surface import plot_3d_surface

__all__ = ["plot_convergence", "plot_2d_contour", "plot_3d_surface"]