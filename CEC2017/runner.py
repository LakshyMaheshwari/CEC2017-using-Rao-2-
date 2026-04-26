import os
import csv
import numpy as np
import random
import time

from tqdm import tqdm

from CEC2017.algorithms import ALGORITHMS
from CEC2017.functions.core import reset_fes, get_fes, get_optimal_value
from CEC2017.visualization.plot_convergence import plot_convergence
from CEC2017.visualization.plot_3d_surface import plot_3d_surface
from CEC2017.visualization.plot_2d_contour import plot_2d_contour
from CEC2017.results import save_results
from CEC2017.config import FES_CHECKPOINTS


# ── Batch comparison CSV collection (FIX 5) ──
_comparison_rows = []  # Collect rows across all run_experiment() calls


def reset_comparison_rows():
    """Reset the collection for a fresh batch."""
    global _comparison_rows
    _comparison_rows = []


def add_comparison_row(row_dict):
    """Add a single comparison row to the batch."""
    global _comparison_rows
    _comparison_rows.append(row_dict)


def write_comparison_csv():
    """Write all collected rows to CSV at once."""
    global _comparison_rows
    if not _comparison_rows:
        return

    csv_path = "results/comparison_summary.csv"
    os.makedirs("results", exist_ok=True)

    header = ["Algorithm", "FuncID", "Dimension", "Best_Error",
              "Worst_Error", "Median_Error", "Mean_Error", "Std_Dev", "Time_s"]

    # Read existing rows (if any)
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader) if reader.fieldnames else []

    # Merge: replace old rows with same key, keep new ones
    keyed = {(r["Algorithm"], r["FuncID"], r["Dimension"]): r
             for r in existing_rows}
    for row in _comparison_rows:
        key = (row["Algorithm"], row["FuncID"], row["Dimension"])
        keyed[key] = row

    # Write all
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for key in sorted(keyed.keys()):
            writer.writerow(keyed[key])

    _comparison_rows = []  # Clear for next batch


def _extract_errors_at_checkpoints(history, max_fes, f_star):
    """
    Given a history of (fes_count, best_fitness) tuples,
    return error values (fitness - Fi*) at the 14 official checkpoints.

    Uses searchsorted — correctly handles all edge cases without
    breaking early or missing the last valid entry before a checkpoint.
    """
    checkpoint_fes = [frac * max_fes for frac in FES_CHECKPOINTS]

    fes_arr = np.array([fes for fes, _ in history], dtype=float)
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


def _prepare_comparison_row(algo_name, func_id, dimension, final_arr, total_time):
    """Prepare (don't write) one CSV row for comparison summary."""
    return {
        "Algorithm": algo_name,
        "FuncID": func_id,
        "Dimension": dimension,
        "Best_Error": f"{np.min(final_arr):.6e}",
        "Worst_Error": f"{np.max(final_arr):.6e}",
        "Median_Error": f"{np.median(final_arr):.6e}",
        "Mean_Error": f"{np.mean(final_arr):.6e}",
        "Std_Dev": f"{np.std(final_arr):.6e}",
        "Time_s": f"{total_time:.2f}",
    }


def run_experiment(algo_name, func_id, dimension, lb, ub, pop_size, max_fes, runs):

    assert algo_name in ALGORITHMS, f"Unknown algorithm: {algo_name}"

    algorithm = ALGORITHMS[algo_name]
    f_star = get_optimal_value(func_id)

    all_histories = []           # raw (fes, fitness) histories per run
    all_checkpoint_errors = []   # 14 error values per run
    final_errors = []
    best_solution = None
    best_value = float("inf")
    best_solutions = []          # Track best solutions for each run

    # ⏱️ Start timer
    start_time = time.time()

    pbar = tqdm(range(runs), desc=f"{algo_name.upper()} F{func_id} D{dimension}")
    for run in pbar:

        random.seed(run)
        np.random.seed(run)

        # Reset FES counter for each independent run
        reset_fes()

        best, history = algorithm(pop_size, dimension, lb, ub, max_fes, func_id)

        fes_used = get_fes()
        # Use the last recorded best fitness from the history
        # instead of calling evaluate() again (which would waste 1 FES)
        _, last_best_f = history[-1]
        error = max(last_best_f - f_star, 0.0)

        # Update progress bar in-place (single clean line)
        pbar.set_postfix(FES=fes_used, err=f"{error:.2e}")

        all_histories.append(history)
        checkpoint_errors = _extract_errors_at_checkpoints(history, max_fes, f_star)
        all_checkpoint_errors.append(checkpoint_errors)
        final_errors.append(error)

        if last_best_f < best_value:
            best_value = last_best_f
            best_solution = best
        best_solutions.append(best.copy())

    end_time = time.time()
    total_time = end_time - start_time

    # Build 14 x RUNS error matrix
    error_matrix = np.array(all_checkpoint_errors).T  # shape: (14, runs)

    # Summary statistics on final errors (last checkpoint)
    final_arr = np.array(final_errors)
    stats = {
        "Best Value": best_value,
        "Best Error": np.min(final_arr),
        "Worst Error": np.max(final_arr),
        "Median Error": np.median(final_arr),
        "Mean Error": np.mean(final_arr),
        "Std Dev": np.std(final_arr),
        "Std Error": np.std(final_arr) / np.sqrt(len(final_arr)),
    }

    # ── Save results with algo-specific paths ──
    save_results(func_id, dimension, error_matrix, stats, total_time,
                 best_solution, best_solutions, runs, algo_name=algo_name)
    plot_convergence(all_histories, func_id, dimension, f_star, algo_name=algo_name)

    if dimension == 2:
        plot_3d_surface(func_id, best_solution, lb, ub, algo_name=algo_name)
        plot_2d_contour(func_id, best_solution, lb, ub, algo_name=algo_name)

    # ── Collect comparison row (batch write at end) — FIX 5 ──
    row = _prepare_comparison_row(algo_name, func_id, dimension, final_arr, total_time)
    add_comparison_row(row)

    print(f"\nF{func_id} | D={dimension} | MaxFES={max_fes} | Algorithm={algo_name.upper()}")
    print(f"Best Value: {stats['Best Value']:.6e}")
    print(f"Best Error: {stats['Best Error']:.6e}")
    print(f"Worst Err : {stats['Worst Error']:.6e}")
    print(f"Median Err: {stats['Median Error']:.6e}")
    print(f"Mean Err  : {stats['Mean Error']:.6e}")
    print(f"Std Dev   : {stats['Std Dev']:.6e}")
    print(f"Std Error : {stats['Std Error']:.6e}")
    print(f"Time      : {total_time:.2f} seconds")
