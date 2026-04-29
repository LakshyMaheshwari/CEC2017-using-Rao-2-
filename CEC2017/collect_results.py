"""
Result Collection Script — produces CSV summary files after experiments complete.

Reads result .txt and _solution.txt files from the results/ directory and
generates:
  1. Per-algorithm summary CSV (one row per function per dimension)
  2. Decision variables CSV (best solution vectors)

Usage:
    python -m CEC2017.collect_results
    python -m CEC2017.collect_results --algo rao2
    python -m CEC2017.collect_results --algo rao2 --dim 10
"""

import os
import re
import csv
import argparse
import numpy as np


# ── Keys recognised in result files ──
STAT_KEYS = {
    "Best Fitness", "Mean Fitness", "Worst Fitness",
    "Std Dev", "SEM", "Ideal", "Runs", "Max FES", "Iterations",
    "Total Time", "Avg Time", "Success Rate",
    # Legacy keys (from old error-based format)
    "Best Value", "Best Error", "Worst Error", "Median Error",
    "Mean Error", "Std Error", "Time",
}


def parse_result_file(filepath):
    """Parse a single result .txt file, return dict of summary stats."""
    stats = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Per-run") or line.startswith("Run "):
                    continue
                parts = line.split("\t")
                if len(parts) == 2 and parts[0] in STAT_KEYS:
                    stats[parts[0]] = parts[1]
    except FileNotFoundError:
        pass
    return stats


def parse_solution_file(filepath):
    """Parse a _solution.txt file, return list of (var_name, value) tuples."""
    variables = []
    try:
        with open(filepath, "r") as f:
            in_vars = False
            for line in f:
                line = line.strip()
                if line == "Decision variables:":
                    in_vars = True
                    continue
                if in_vars and line:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        variables.append((parts[0], parts[1]))
    except FileNotFoundError:
        pass
    return variables


def _discover_algorithms(results_dir):
    """Auto-detect algorithm subdirectories in the results folder."""
    algos = []
    if not os.path.isdir(results_dir):
        return algos
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and not entry.startswith(("F", ".", "_")):
            algos.append(entry)
    return algos


def _detect_dimensions(results_dir, algo):
    """Auto-detect all dimensions from existing result files for an algorithm."""
    dims = set()
    pattern = re.compile(r"_D(\d+)\.txt$")
    algo_dir = os.path.join(results_dir, algo)
    if not os.path.isdir(algo_dir):
        return sorted(dims)
    for func_dir in os.listdir(algo_dir):
        func_path = os.path.join(algo_dir, func_dir)
        if not os.path.isdir(func_path):
            continue
        for fname in os.listdir(func_path):
            if "_solution" in fname:
                continue
            m = pattern.search(fname)
            if m:
                dims.add(int(m.group(1)))
    return sorted(dims)


def collect_summary_csv(results_dir, algo, dim, output_dir=None):
    """
    Produce a per-function summary CSV for one algorithm + dimension.

    Columns: Function No., I, B, M, W, SD, SEM, Time, Iteration, FE, Speedup, Success Rate
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, algo)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"summary_{algo}_D{dim}.csv")

    # Collect all run times for speedup calculation (reference = max time)
    all_times = []
    rows_data = []

    for func_id in range(1, 31):
        if func_id == 2:
            continue

        filepath = os.path.join(
            results_dir, algo, f"F{func_id}",
            f"{algo}_F{func_id}_D{dim}.txt"
        )
        stats = parse_result_file(filepath)

        if not stats:
            continue

        # Extract values — handle both new fitness-based and legacy error-based formats
        ideal = _safe_float(stats.get("Ideal", ""))
        best = _safe_float(stats.get("Best Fitness", stats.get("Best Value", "")))
        mean = _safe_float(stats.get("Mean Fitness", stats.get("Mean Error", "")))
        worst = _safe_float(stats.get("Worst Fitness", stats.get("Worst Error", "")))
        sd = _safe_float(stats.get("Std Dev", ""))
        sem = _safe_float(stats.get("SEM", stats.get("Std Error", "")))
        avg_time = _safe_float(stats.get("Avg Time", stats.get("Time", "")))
        runs = stats.get("Runs", "")
        max_fes = stats.get("Max FES", "")
        iterations = stats.get("Iterations", "")
        success_rate = stats.get("Success Rate", "")

        if avg_time is not None:
            all_times.append(avg_time)

        # Compute total iterations = runs × iterations_per_run
        runs_int = int(runs) if runs else 0
        iter_int = int(iterations) if iterations else 0
        total_iterations = runs_int * iter_int

        rows_data.append({
            "func_id": func_id,
            "ideal": ideal,
            "best": best,
            "mean": mean,
            "worst": worst,
            "sd": sd,
            "sem": sem,
            "avg_time": avg_time,
            "runs": runs,
            "iterations": iterations,
            "total_iterations": total_iterations,
            "max_fes": max_fes,
            "success_rate": success_rate,
        })

    if not rows_data:
        print(f"  No results found for {algo} D{dim}")
        return

    # Compute reference time for speedup (maximum time across functions)
    ref_time = max(all_times) if all_times else 1.0

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Function No.", "I", "B", "M", "W",
            "SD", "SEM", "Time", "Runs", "Iteration",
            "Total Iterations", "FE", "Speedup", "Success Rate"
        ])

        for row in rows_data:
            speedup = ""
            if row["avg_time"] is not None and row["avg_time"] > 0:
                speedup = f"{ref_time / row['avg_time']:.4f}"

            writer.writerow([
                f"F{row['func_id']}",
                _fmt(row["ideal"]),
                _fmt(row["best"]),
                _fmt(row["mean"]),
                _fmt(row["worst"]),
                _fmt(row["sd"]),
                _fmt(row["sem"]),
                f"{row['avg_time']:.2f}" if row["avg_time"] is not None else "—",
                row["runs"],
                row["iterations"],
                row["total_iterations"],
                row["max_fes"],
                speedup,
                f"{row['success_rate']}%" if row["success_rate"] else "—",
            ])

    print(f"  Summary CSV saved: {output_path}")


def collect_decision_vars_csv(results_dir, algo, dim, output_dir=None):
    """
    Produce a decision variables CSV for one algorithm + dimension.
    Each row = best solution vector for one function.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, algo)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"decision_vars_{algo}_D{dim}.csv")

    # Determine max number of variables
    max_vars = dim
    header = ["Function"] + [f"x{i+1}" for i in range(max_vars)]

    rows = []
    for func_id in range(1, 31):
        if func_id == 2:
            continue

        solution_path = os.path.join(
            results_dir, algo, f"F{func_id}",
            f"{algo}_F{func_id}_D{dim}_solution.txt"
        )
        variables = parse_solution_file(solution_path)

        if not variables:
            continue

        row = [f"F{func_id}"]
        for _, val in variables:
            row.append(val)
        # Pad if fewer variables than expected
        while len(row) < len(header):
            row.append("")

        rows.append(row)

    if not rows:
        print(f"  No solution files found for {algo} D{dim}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Decision vars CSV saved: {output_path}")


def _safe_float(val):
    """Safely convert to float, return None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt(val):
    """Format a float for CSV output, or return dash."""
    if val is None:
        return "—"
    return f"{val:.6e}"


def main():
    parser = argparse.ArgumentParser(
        description="Collect CEC2017 benchmark results into CSV files."
    )
    parser.add_argument("--algo", type=str, default=None,
                        help="Algorithm to collect (default: all detected)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Dimension to collect (default: all detected)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to results directory (default: results)")

    args = parser.parse_args()

    results_dir = args.results_dir

    # Discover algorithms
    if args.algo:
        algos = [args.algo]
    else:
        algos = _discover_algorithms(results_dir)

    if not algos:
        print("No algorithm directories found in results/. Nothing to collect.")
        return

    print(f"Collecting results for algorithms: {algos}")
    print("=" * 60)

    for algo in algos:
        print(f"\n  Algorithm: {algo.upper()}")

        # Discover dimensions
        if args.dim:
            dims = [args.dim]
        else:
            dims = _detect_dimensions(results_dir, algo)

        if not dims:
            print(f"    No result files found for {algo}")
            continue

        print(f"    Dimensions: {dims}")

        for dim in dims:
            print(f"\n    ─── D={dim} ───")
            collect_summary_csv(results_dir, algo, dim)
            collect_decision_vars_csv(results_dir, algo, dim)

    print("\n" + "=" * 60)
    print("  Result collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
