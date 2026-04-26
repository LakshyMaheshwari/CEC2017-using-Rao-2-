import warnings
from functools import lru_cache

from .cec2017.all_functions import ALL_FUNCTIONS


@lru_cache(maxsize=32)
def get_function(func_id: int) -> dict:
    """
    Get the objective function for a given CEC2017 function ID.

    Parameters
    ----------
    func_id : int
        Function ID in [1, 30]. Note: F2 is deprecated in CEC2017.

    Returns
    -------
    dict
        Dictionary with keys: "objective", "name", "optimal_value".
    """
    if func_id == 2:
        warnings.warn(
            "F2 (Schwefel's function) is deprecated in CEC2017. "
            "Consider using F1 or other functions for benchmarking.",
            DeprecationWarning,
            stacklevel=2,
        )

    if 1 <= func_id <= 30:
        return ALL_FUNCTIONS[func_id]
    else:
        raise ValueError(f"Function {func_id} not implemented")
