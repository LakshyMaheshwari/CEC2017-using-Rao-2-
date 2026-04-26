import numpy as np


def apply_bounds(x, lb, ub):
    return np.clip(x, lb, ub)
