"""
Smoke and correctness tests for CEC2017 algorithms.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from CEC2017.algorithms import ALGORITHMS
from CEC2017.functions.core import reset_fes, get_fes, fes_counter, evaluate
from CEC2017.utils.bounds import apply_bounds


class TestBounds:
    """Test boundary reflection."""

    def test_reflect_below_lower(self):
        """Test reflection below lower bound."""
        x = np.array([-150.0, 50.0, 100.0])
        result = apply_bounds(x, -100, 100)
        assert np.all(result >= -100) and np.all(result <= 100), \
            f"Out of bounds: {result}"

    def test_reflect_above_upper(self):
        """Test reflection above upper bound."""
        x = np.array([50.0, 150.0, 200.0])
        result = apply_bounds(x, -100, 100)
        assert np.all(result >= -100) and np.all(result <= 100), \
            f"Out of bounds: {result}"

    def test_in_bounds_unchanged(self):
        """Test that in-bounds values stay unchanged (approx)."""
        x = np.array([-50.0, 0.0, 50.0])
        result = apply_bounds(x, -100, 100)
        np.testing.assert_array_almost_equal(result, x)

    def test_reflect_symmetry(self):
        """Reflection of -150 at lb=-100 should give -50."""
        x = np.array([-150.0])
        result = apply_bounds(x, -100, 100)
        np.testing.assert_almost_equal(result[0], -50.0)

    def test_reflect_upper_symmetry(self):
        """Reflection of 150 at ub=100 should give 50."""
        x = np.array([150.0])
        result = apply_bounds(x, -100, 100)
        np.testing.assert_almost_equal(result[0], 50.0)


class TestFESCounter:
    """Test function evaluation counter."""

    def test_counter_starts_at_zero(self):
        """Counter should be 0 after reset."""
        reset_fes()
        assert get_fes() == 0

    def test_counter_increments(self):
        """Test that counter increments correctly."""
        reset_fes()
        fes_counter.increment()
        assert get_fes() == 1

    def test_counter_reset(self):
        """Test that counter resets."""
        fes_counter.increment()
        fes_counter.increment()
        reset_fes()
        assert get_fes() == 0


class TestFunctionEvaluation:
    """Test CEC2017 function evaluation."""

    def test_f1_returns_scalar(self):
        """evaluate() should return (float, 0.0)."""
        reset_fes()
        x = np.zeros(2)
        f, constraint = evaluate(x, func_id=1)

        assert isinstance(f, (int, float, np.floating)), \
            f"f should be scalar, got {type(f)}"
        assert constraint == 0.0, \
            "CEC2017 unconstrained, constraint should be 0"
        assert not np.isnan(f), "fitness should not be NaN"
        assert not np.isinf(f), "fitness should not be inf"

    def test_evaluate_increments_fes(self):
        """Each evaluate() call should increment FES by exactly 1."""
        reset_fes()
        x = np.zeros(2)
        evaluate(x, func_id=1)
        assert get_fes() == 1
        evaluate(x, func_id=1)
        assert get_fes() == 2

    def test_f2_deprecation_warning(self):
        """F2 should emit a DeprecationWarning."""
        reset_fes()
        with pytest.warns(DeprecationWarning, match="F2.*deprecated"):
            evaluate(np.zeros(2), func_id=2)


class TestAlgorithmsSmoke:
    """Smoke test: all algorithms should complete without error."""

    @pytest.mark.parametrize("algo_name", list(ALGORITHMS.keys()))
    def test_algorithm_runs(self, algo_name):
        """Run each algorithm on F1, D=2, for 50 FES."""
        reset_fes()
        algo = ALGORITHMS[algo_name]
        best, history = algo(
            pop_size=5,
            D=2,
            lb=-100,
            ub=100,
            max_fes=50,
            func_id=1,
        )

        assert isinstance(best, np.ndarray), \
            f"{algo_name}: best should be ndarray"
        assert best.shape == (2,), \
            f"{algo_name}: best shape should be (2,), got {best.shape}"
        assert isinstance(history, list), \
            f"{algo_name}: history should be list"
        assert len(history) > 0, \
            f"{algo_name}: history should not be empty"
        assert get_fes() <= 50, \
            f"{algo_name}: FES exceeded budget: {get_fes()}"

    def test_rao2_convergence_monotonic(self):
        """Rao-2 convergence history should be monotonically non-increasing."""
        reset_fes()
        rao2 = ALGORITHMS["rao2"]
        best, history = rao2(
            pop_size=10,
            D=2,
            lb=-100,
            ub=100,
            max_fes=100,
            func_id=1,
        )

        fitness_values = [f for _, f in history]
        for i in range(1, len(fitness_values)):
            assert fitness_values[i] <= fitness_values[i - 1] + 1e-10, \
                f"History not monotonic at index {i}: " \
                f"{fitness_values[i]} > {fitness_values[i-1]}"

    def test_bounds_respected(self):
        """All algorithms should keep solutions within bounds."""
        for algo_name in ALGORITHMS:
            reset_fes()
            algo = ALGORITHMS[algo_name]
            best, _ = algo(
                pop_size=5,
                D=2,
                lb=-100,
                ub=100,
                max_fes=50,
                func_id=1,
            )
            assert np.all(best >= -100) and np.all(best <= 100), \
                f"{algo_name}: solution out of bounds: {best}"
