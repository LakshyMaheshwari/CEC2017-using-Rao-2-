import os
import csv
import sys
import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from CEC2017.algorithms import ALGORITHMS
from CEC2017.functions.core import reset_fes, get_fes, get_optimal_value
from CEC2017.visualization.plot_convergence import plot_convergence
from CEC2017.visualization.plot_3d_surface import plot_3d_surface
from CEC2017.visualization.plot_2d_contour import plot_2d_contour
from CEC2017.results import save_results
from CEC2017.config import MAX_ITERATIONS


# ── CSV header for per-dimension summary files ──
_SUMMARY_CSV_HEADER = [
    "Function No.", "I", "B", "M", "W",
    "SD", "SEM", "Time", "Runs", "Iteration",
    "Total Iterations", "FE", "Speedup", "Success Rate"
]


def append_to_summary_csv(algo_name, func_id, dimension, stats,
                          run_times, runs, max_fes, success_rate):
    """
    Append one row to results/{algo}/summary_{algo}_D{dim}.csv
    immediately after a function completes.
    Creates the file with headers if it doesn't exist.
    """
    folder = f"results/{algo_name}"
    os.makedirs(folder, exist_ok=True)
    csv_path = f"{folder}/summary_{algo_name}_D{dimension}.csv"

    file_exists = os.path.exists(csv_path)

    # Read existing rows to compute speedup reference & avoid duplicates
    existing_rows = []
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                existing_rows = [r for r in reader if r.get("Function No.") != f"F{func_id}"]

    avg_time = np.mean(run_times)
    iterations = MAX_ITERATIONS
    total_iterations = runs * iterations

    new_row = {
        "Function No.": f"F{func_id}",
        "I": f"{stats['Ideal']:.6e}",
        "B": f"{stats['Best Fitness']:.6e}",
        "M": f"{stats['Mean Fitness']:.6e}",
        "W": f"{stats['Worst Fitness']:.6e}",
        "SD": f"{stats['Std Dev']:.6e}",
        "SEM": f"{stats['SEM']:.6e}",
        "Time": f"{avg_time:.2f}",
        "Runs": str(runs),
        "Iteration": str(iterations),
        "Total Iterations": str(total_iterations),
        "FE": str(max_fes),
        "Speedup": "",
        "Success Rate": f"{success_rate:.1f}%",
    }

    all_rows = existing_rows + [new_row]

    # Sort by function number
    def _func_sort_key(r):
        try:
            return int(r["Function No."].replace("F", ""))
        except (ValueError, KeyError):
            return 999
    all_rows.sort(key=_func_sort_key)

    # Compute speedup (reference = max time across all functions)
    times = []
    for r in all_rows:
        try:
            times.append(float(r["Time"]))
        except (ValueError, KeyError):
            pass
    ref_time = max(times) if times else 1.0
    for r in all_rows:
        try:
            t = float(r["Time"])
            r["Speedup"] = f"{ref_time / t:.4f}" if t > 0 else ""
        except (ValueError, KeyError):
            r["Speedup"] = ""

    # Write entire file (sorted, with updated speedups)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Summary CSV updated: {csv_path}")


# ── Batch comparison CSV collection ──
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

    header = ["Algorithm", "FuncID", "Dimension", "Ideal",
              "Best_Fitness", "Mean_Fitness", "Worst_Fitness",
              "Std_Dev", "SEM", "Avg_Time_s", "Runs", "Max_FES",
              "Success_Rate"]

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
        for key in sorted(keyed.keys(), key=lambda k: (str(k[0]), int(k[1]), int(k[2]))):
            writer.writerow(keyed[key])

    _comparison_rows = []  # Clear for next batch


def _prepare_comparison_row(algo_name, func_id, dimension, final_fitness_arr,
                            run_times, f_star, max_fes, runs, success_count):
    """Prepare one CSV row for comparison summary (fitness-based)."""
    return {
        "Algorithm": algo_name,
        "FuncID": func_id,
        "Dimension": dimension,
        "Ideal": f"{f_star:.6e}",
        "Best_Fitness": f"{np.min(final_fitness_arr):.6e}",
        "Mean_Fitness": f"{np.mean(final_fitness_arr):.6e}",
        "Worst_Fitness": f"{np.max(final_fitness_arr):.6e}",
        "Std_Dev": f"{np.std(final_fitness_arr):.6e}",
        "SEM": f"{np.std(final_fitness_arr) / np.sqrt(len(final_fitness_arr)):.6e}",
        "Avg_Time_s": f"{np.mean(run_times):.2f}",
        "Runs": runs,
        "Max_FES": max_fes,
        "Success_Rate": f"{(success_count / runs) * 100:.1f}%",
    }


# ── Per-run detailed CSV ──

def save_per_run_csv(algo_name, func_id, dimension, results_list):
    """
    Save a CSV with one row per run containing:
    Run, x1, x2, ..., xD, F(x), FES_Used, Time(s), Success

    Path: results/{algo}/F{id}/{algo}_F{id}_D{dim}_all_runs.csv
    """
    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_F{func_id}"
    else:
        folder = f"results/F{func_id}"
        prefix = f"F{func_id}"
    os.makedirs(folder, exist_ok=True)

    csv_path = f"{folder}/{prefix}_D{dimension}_all_runs.csv"

    D = len(results_list[0]["best"])
    header = ["Run"] + [f"x{i+1}" for i in range(D)] + ["F(x)", "FES_Used", "Time(s)", "Success"]

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in results_list:
                row = [r["run_id"] + 1]                         # 1-indexed run number
                row += [f"{v:.10e}" for v in r["best"]]         # decision variables
                row.append(f"{r['last_best_f']:.10e}")          # F(x)
                row.append(r["fes_used"])                       # FES used
                row.append(f"{r['run_time']:.4f}")              # time in seconds
                row.append(r["success"])                        # success flag
                writer.writerow(row)
        print(f"Per-run CSV saved: {csv_path}")
    except PermissionError:
        print(f"\n[WARNING] Permission denied to write {csv_path}. Is it open in Excel?")


def save_convergence_csv(algo_name, func_id, dimension, results_list):
    """
    Save the complete convergence history for every run.
    Each row = one recorded iteration from one run.

    Columns: Run, FES, Best_F(x)
    Path: results/{algo}/F{id}/{algo}_F{id}_D{dim}_convergence.csv
    """
    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_F{func_id}"
    else:
        folder = f"results/F{func_id}"
        prefix = f"F{func_id}"
    os.makedirs(folder, exist_ok=True)

    csv_path = f"{folder}/{prefix}_D{dimension}_convergence.csv"

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "FES", "Best_F(x)"])
            for r in results_list:
                run_num = r["run_id"] + 1
                for fes, best_f in r["history"]:
                    writer.writerow([run_num, fes, f"{best_f:.10e}"])
        print(f"Convergence CSV saved: {csv_path}")
    except PermissionError:
        print(f"\n[WARNING] Permission denied to write {csv_path}. Is it open in Excel?")


def append_to_master_summary_csv(algo_name, func_id, dimension, best_solution, best_fitness, f_star):
    """
    Append one row (best solution for this function) to a master cross-function
    summary CSV. One file per algo+dimension, accumulates across all functions.

    Columns: Function, x1, x2, ..., xD, F(x), Ideal
    Path: results/{algo}/master_summary_{algo}_D{dim}.csv
    """
    folder = f"results/{algo_name}" if algo_name else "results"
    os.makedirs(folder, exist_ok=True)
    prefix = algo_name if algo_name else "results"
    csv_path = f"{folder}/master_summary_{prefix}_D{dimension}.csv"

    D = len(best_solution)
    header = ["Function"] + [f"x{i+1}" for i in range(D)] + ["F(x)", "Ideal"]

    # Read existing rows, replace if same function already present
    existing_rows = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_rows = [r for r in reader if r.get("Function") != f"F{func_id}"]
        except Exception:
            existing_rows = []

    new_row = {"Function": f"F{func_id}"}
    for i, v in enumerate(best_solution):
        new_row[f"x{i+1}"] = f"{v:.10e}"
    new_row["F(x)"] = f"{best_fitness:.10e}"
    new_row["Ideal"] = f"{f_star:.10e}"

    all_rows = existing_rows + [new_row]

    # Sort by function number
    def _sort_key(r):
        try:
            return int(r["Function"].replace("F", ""))
        except (ValueError, KeyError):
            return 999
    all_rows.sort(key=_sort_key)

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Master summary CSV updated: {csv_path}")
    except PermissionError:
        print(f"\n[WARNING] Permission denied to write {csv_path}. Is it open in Excel?")


# ── Multiprocessing worker ──

def _run_single(args):
    """
    Worker function for a single independent run.
    Runs in a separate process — each has its own FES counter.
    Returns: dict with run results.
    """
    run_id, algo_name, pop_size, dimension, lb, ub, max_fes, func_id, f_star = args

    # Each process gets its own RNG state
    random.seed(run_id)
    np.random.seed(run_id)

    # Each process has its own FES counter (module-level singleton per process)
    reset_fes()

    algorithm = ALGORITHMS[algo_name]

    run_start = time.time()

    best, history = algorithm(
        pop_size, dimension, lb, ub, max_fes, func_id,
        early_stop_value=f_star,
    )

    run_time = time.time() - run_start
    fes_used = get_fes()

    _, last_best_f = history[-1]
    success = abs(last_best_f - f_star) < 1e-2

    return {
        "run_id": run_id,
        "best": best,
        "history": history,
        "last_best_f": last_best_f,
        "run_time": run_time,
        "fes_used": fes_used,
        "success": success,
    }


def run_experiment(
    algo_name: str,
    func_id: int,
    dimension: int,
    lb: float,
    ub: float,
    pop_size: int,
    max_fes: int,
    runs: int,
    n_workers: int = None,
) -> None:
    """
    Run optimization experiment: N independent runs (parallelized),
    report actual fitness values.

    n_workers: number of parallel processes (default: CPU count - 1, min 1).
    """

    # ── Input validation ──
    if algo_name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algo_name}. "
            f"Choose from: {list(ALGORITHMS.keys())}"
        )

    if func_id not in range(1, 31) or func_id == 2:
        import warnings
        if func_id == 2:
            warnings.warn(
                f"F2 is deprecated in CEC2017. Results may be unreliable.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif func_id not in range(1, 31):
            raise ValueError(
                f"Function ID must be in [1, 30]. Got {func_id}"
            )

    if dimension not in [2, 10, 20, 30, 50, 100]:
        import warnings
        warnings.warn(
            f"Dimension {dimension} is unusual. "
            f"CEC2017 standard uses [2, 10, 20, 30, 50, 100]."
        )

    if pop_size < 2:
        raise ValueError(f"Population size must be >= 2, got {pop_size}")

    if max_fes < pop_size:
        raise ValueError(
            f"max_fes ({max_fes}) must be >= pop_size ({pop_size}). "
            f"Otherwise, first generation cannot complete."
        )

    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    f_star = get_optimal_value(func_id)

    # ── Determine worker count ──
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    n_workers = min(n_workers, runs)  # No point having more workers than runs

    # Prepare worker arguments
    worker_args = [
        (run_id, algo_name, pop_size, dimension, lb, ub, max_fes, func_id, f_star)
        for run_id in range(runs)
    ]

    # Start timer
    total_start = time.time()

    # ── Execute runs (parallel or sequential) ──
    results_list = [None] * runs

    if n_workers > 1:
        # Parallel execution
        print(f"  Using {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single, args): args[0]
                       for args in worker_args}

            pbar = tqdm(total=runs, desc=f"{algo_name.upper()} F{func_id} D{dimension}")
            for future in as_completed(futures):
                result = future.result()
                run_id = result["run_id"]
                results_list[run_id] = result
                pbar.set_postfix(
                    FES=result["fes_used"],
                    fitness=f"{result['last_best_f']:.2e}"
                )
                pbar.update(1)
            pbar.close()
            sys.stdout.flush()
            sys.stderr.flush()
    else:
        # Sequential fallback (1 worker)
        pbar = tqdm(range(runs), desc=f"{algo_name.upper()} F{func_id} D{dimension}")
        for run_id in pbar:
            result = _run_single(worker_args[run_id])
            results_list[run_id] = result
            pbar.set_postfix(
                FES=result["fes_used"],
                fitness=f"{result['last_best_f']:.2e}"
            )
        pbar.close()
        sys.stdout.flush()
        sys.stderr.flush()

    total_end = time.time()
    total_time = total_end - total_start

    # ── Aggregate results ──
    all_histories = [r["history"] for r in results_list]
    final_fitness_values = [r["last_best_f"] for r in results_list]
    run_times = [r["run_time"] for r in results_list]
    fes_used_list = [r["fes_used"] for r in results_list]
    best_solutions = [r["best"].copy() for r in results_list]
    success_count = sum(1 for r in results_list if r["success"])

    # Find overall best
    best_idx = int(np.argmin(final_fitness_values))
    best_solution = results_list[best_idx]["best"]

    # Summary statistics on actual fitness values
    final_arr = np.array(final_fitness_values)
    stats = {
        "Best Fitness": np.min(final_arr),
        "Mean Fitness": np.mean(final_arr),
        "Worst Fitness": np.max(final_arr),
        "Std Dev": np.std(final_arr),
        "SEM": np.std(final_arr) / np.sqrt(len(final_arr)),
        "Ideal": f_star,
        "Success Rate": (success_count / runs) * 100,
    }

    # ── Print per-run data ──
    print(f"\n{'─'*80}")
    print(f"  Per-Run Results: {algo_name.upper()} F{func_id} D{dimension}")
    print(f"{'─'*80}")
    print(f"  {'Run':<5} {'F(x)':<22} {'FES Used':<12} {'Time(s)':<10} {'Success'}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*7}")
    for r in results_list:
        print(f"  {r['run_id']+1:<5} {r['last_best_f']:<22.10e} {r['fes_used']:<12} {r['run_time']:<10.4f} {r['success']}")
    print(f"{'─'*80}")

    # ── Save results with algo-specific paths ──
    save_results(func_id, dimension, stats, total_time, run_times,
                 best_solution, best_solutions, runs, max_fes,
                 fes_used_list, algo_name=algo_name, best_run_id=best_idx)

    # ── Save per-run detailed CSV and convergence history ──
    save_per_run_csv(algo_name, func_id, dimension, results_list)
    save_convergence_csv(algo_name, func_id, dimension, results_list)

    # ── Append best solution to master cross-function summary ──
    append_to_master_summary_csv(
        algo_name, func_id, dimension,
        best_solution, stats['Best Fitness'], f_star
    )

    plot_convergence(all_histories, func_id, dimension, f_star, algo_name=algo_name)

    if dimension == 2:
        plot_3d_surface(func_id, best_solution, lb, ub, algo_name=algo_name)
        plot_2d_contour(func_id, best_solution, lb, ub, algo_name=algo_name)

    # ── Collect comparison row (batch write at end) ──
    row = _prepare_comparison_row(
        algo_name, func_id, dimension, final_arr,
        run_times, f_star, max_fes, runs, success_count
    )
    add_comparison_row(row)

    # ── Append to per-dimension summary CSV immediately ──
    append_to_summary_csv(
        algo_name, func_id, dimension, stats,
        run_times, runs, max_fes, stats['Success Rate']
    )

    print(f"\nF{func_id} | D={dimension} | MaxFES={max_fes} | Algorithm={algo_name.upper()}")
    print(f"Ideal        : {f_star:.6e}")
    print(f"Best Fitness : {stats['Best Fitness']:.6e}")
    print(f"Mean Fitness : {stats['Mean Fitness']:.6e}")
    print(f"Worst Fitness: {stats['Worst Fitness']:.6e}")
    print(f"Std Dev      : {stats['Std Dev']:.6e}")
    print(f"SEM          : {stats['SEM']:.6e}")
    print(f"Success Rate : {stats['Success Rate']:.1f}%")
    print(f"Avg Time/Run : {np.mean(run_times):.2f} seconds")
    print(f"Total Time   : {total_time:.2f} seconds ({n_workers} workers)")
