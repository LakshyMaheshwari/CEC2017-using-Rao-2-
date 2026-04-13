import numpy as np
import random
import time

from .algorithms.rao2 import rao2
from .functions.core import evaluate, reset_fes, get_fes, get_optimal_value
from .visualization.plot_convergence import plot_convergence
from .visualization.plot_3d_surface import plot_3d_surface
from .visualization.plot_2d_contour import plot_2d_contour
from .results import save_results
from .config import FES_CHECKPOINTS


def _extract_errors_at_checkpoints(history, max_fes, f_star):
    """
    Given a history of (fes_count, best_fitness) tuples,
    return error values (fitness - Fi*) at the 14 official checkpoints.

    Uses searchsorted — correctly handles all edge cases without
    breaking early or missing the last valid entry before a checkpoint.
    """
    checkpoint_fes = [frac * max_fes for frac in FES_CHECKPOINTS]

    fes_arr = np.array([fes for fes, _   in history], dtype=float)
    fit_arr = np.array([fit for _,   fit in history], dtype=float)

    errors = []
    for cp_fes in checkpoint_fes:
        # searchsorted finds insertion point — subtract 1 to get
        # the last recorded entry AT OR BEFORE this checkpoint
        idx = np.searchsorted(fes_arr, cp_fes, side='right') - 1

        if idx < 0:
            # No evaluations recorded before this checkpoint yet
            # Use the very first recorded fitness as fallback
            fit_at_cp = fit_arr[0]
        else:
            fit_at_cp = fit_arr[idx]

        error = max(fit_at_cp - f_star, 0.0)
        # Clamp to zero per official spec: errors < 1e-8 count as zero
        if error < 1e-8:
            error = 0.0
        errors.append(error)

    return errors

def run_experiment(func_id, dimension, lb, ub, pop_size, max_fes, runs):

    f_star = get_optimal_value(func_id)

    all_histories = []           # raw (fes, fitness) histories per run
    all_checkpoint_errors = []   # 14 error values per run
    final_errors = []
    best_solution = None
    best_value = float("inf")

    # ⏱️ Start timer
    start_time = time.time()

    for run in range(runs):

        random.seed(run)
        np.random.seed(run)

        # Reset FES counter for each independent run
        reset_fes()

        best, history = rao2(pop_size, dimension, lb, ub, max_fes, func_id)

        fes_used = get_fes()
        f, _ = evaluate(best, func_id)
        error = max(f - f_star, 0.0)

        all_histories.append(history)
        checkpoint_errors = _extract_errors_at_checkpoints(history, max_fes, f_star)
        all_checkpoint_errors.append(checkpoint_errors)
        final_errors.append(error)

        if f < best_value:
            best_value = f
            best_solution = best

        print(f"Run {run+1:02d} | Error: {error:.6e} | FES used: {fes_used}")

    end_time = time.time()
    total_time = end_time - start_time

    # Build 14 x RUNS error matrix
    error_matrix = np.array(all_checkpoint_errors).T  # shape: (14, runs)

    # Summary statistics on final errors (last checkpoint)
    final_arr = np.array(final_errors)
    stats = {
        "Best": np.min(final_arr),
        "Worst": np.max(final_arr),
        "Median": np.median(final_arr),
        "Mean": np.mean(final_arr),
        "Std": np.std(final_arr),
    }

    save_results(func_id, dimension, error_matrix, stats, total_time)
    plot_convergence(all_histories, func_id, dimension, f_star)

    if dimension == 2:
        plot_3d_surface(func_id, best_solution, lb, ub)
        plot_2d_contour(func_id, best_solution, lb, ub)

    print(f"\nF{func_id} | D={dimension} | MaxFES={max_fes}")
    print(f"Best  : {stats['Best']:.6e}")
    print(f"Worst : {stats['Worst']:.6e}")
    print(f"Median: {stats['Median']:.6e}")
    print(f"Mean  : {stats['Mean']:.6e}")
    print(f"Std   : {stats['Std']:.6e}")
    print(f"Time  : {total_time:.2f} seconds")