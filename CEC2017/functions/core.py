from functions.get_function import get_function


# ============================================================================
# FES Counter — class-based, parallel safe
# ============================================================================

class FESCounter:
    """
    Encapsulates the function evaluation counter.
    One instance per process — no shared global state.
    Safe for multiprocessing (each worker gets its own instance via
    module reimport or explicit passing).
    """
    def __init__(self):
        self._count = 0

    def increment(self):
        self._count += 1

    def reset(self):
        self._count = 0

    def get(self):
        return self._count

    def set(self, value):
        """Used by visualization to restore counter after plotting."""
        self._count = int(value)


# Module-level singleton — one per process
fes_counter = FESCounter()


# ── Convenience wrappers — all existing call sites stay unchanged ──────────

def reset_fes():
    """Reset FES counter to zero. Call at the start of each run."""
    fes_counter.reset()


def get_fes():
    """Return current number of function evaluations."""
    return fes_counter.get()


# ============================================================================
# Evaluation
# ============================================================================

def get_optimal_value(func_id):
    """
    Return Fi* for CEC2017: Fi* = func_id * 100.
    F2 is deprecated but returns 200 which matches its constant output.
    """
    return func_id * 100


def evaluate(x, func_id):
    """
    Evaluate the objective function and increment the FES counter by 1.

    Returns (objective_value, 0.0).
    The second return value is always 0.0 because CEC2017 F1–F30 are
    unconstrained problems (no inequality or equality constraints).
    It is kept for API compatibility in case constrained functions are
    added in the future.
    """
    fes_counter.increment()

    func = get_function(func_id)
    obj = func["objective"](x)

    return obj, 0.0