import numpy as np


def apply_bounds(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """
    Reflect out-of-bounds dimensions back into search space.

    Uses boundary reflection (not clipping) to comply with CEC2017 standard.
    When a dimension exceeds a boundary, it bounces back as:
      if x_i < lb: x_i = 2*lb - x_i
      if x_i > ub: x_i = 2*ub - x_i
    Repeat until all dimensions are in bounds.

    Parameters
    ----------
    x : np.ndarray
        Candidate solution, shape (D,).
    lb : float
        Lower bound.
    ub : float
        Upper bound.

    Returns
    -------
    np.ndarray
        Reflected solution strictly within [lb, ub].
    """
    x_new = x.copy()
    # Reflect until all dimensions are in bounds
    # (typically 1–2 iterations for most cases)
    max_iters = 10  # safety limit
    for _ in range(max_iters):
        if np.all(x_new >= lb) and np.all(x_new <= ub):
            break
        x_new = np.where(x_new < lb, 2 * lb - x_new, x_new)
        x_new = np.where(x_new > ub, 2 * ub - x_new, x_new)
    return np.clip(x_new, lb, ub)  # Final safety clip
