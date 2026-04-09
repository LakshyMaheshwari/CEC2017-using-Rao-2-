from functions.get_function import get_function

# ── FES (Function Evaluation) counter ────────────────────────────
_fes_counter = 0


def reset_fes():
    """Reset FES counter to zero (call at the start of each run)."""
    global _fes_counter
    _fes_counter = 0


def get_fes():
    """Return the current number of function evaluations."""
    return _fes_counter


def get_optimal_value(func_id):
    """Return Fi* for CEC2017: Fi* = func_id * 100."""
    return func_id * 100


def evaluate(x, func_id):
    global _fes_counter
    _fes_counter += 1

    func = get_function(func_id)

    # Objective
    obj = func["objective"](x)

    # Constraints
    violation = 0.0

    # Inequality constraints g(x) <= 0
    for g in func["g"]:
        val = g(x)
        if val > 0:
            violation += val

    # Equality constraints h(x) = 0
    for h in func["h"]:
        val = h(x)
        if abs(val) > 1e-6:
            violation += abs(val)

    return obj, violation