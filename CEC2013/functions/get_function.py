import warnings
from functools import lru_cache

from .cec2013.all_functions import ALL_FUNCTIONS


@lru_cache(maxsize=32)
def get_function(func_id: int) -> dict:
    """
    Get the objective function for a given CEC2013 function ID.

    Parameters
    ----------
    func_id : int
        Function ID in [1, 28].

    Returns
    -------
    dict
        Dictionary with keys: "objective", "lb", "ub", "g", "h".
    """
    if 1 <= func_id <= 28:
        return ALL_FUNCTIONS[func_id]
    else:
        raise ValueError(f"Function {func_id} not implemented. Must be 1-28.")
