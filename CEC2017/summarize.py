"""
Crawl results/ folder and build a single Big Summary CSV.
Extracts fitness stats (not error) from every result .txt file.

Supports the multi-algorithm directory structure:
    results/{algo}/F{id}/{algo}_F{id}_D{dim}.txt

Usage:
    python -m CEC2017.summarize
"""

import os
import csv
import re
import itertools

import numpy as np
from scipy.stats import ranksums


# Keys we recognise in result files (order matters for CSV columns)
STAT_KEYS = {
    "Best Fitness", "Mean Fitness", "Worst Fitness",
    "Std Dev", "SEM", "Ideal", "Runs", "Max FES", "Iterations",
    "Total Time", "Avg Time", "Success Rate",
    # Legacy keys (from old error-based format) mapped during parsing
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
                    key = parts[0]
                    # Normalise legacy keys to current names
                    if key == "Best Value":
                        key = "Best Fitness"
                    elif key == "Best Error":
                        key = "Best Fitness"
                    elif key == "Worst Error":
                        key = "Worst Fitness"
                    elif key == "Median Error":
                        pass  # drop, not used in new format
                    elif key == "Mean Error":
                        key = "Mean Fitness"
                    elif key == "Std Error":
                        key = "SEM"
                    elif key == "Time":
                        key = "Avg Time"
                    stats[key] = parts[1]
    except FileNotFoundError:
        pass
    return stats


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


def _detect_dimensions(results_dir, algos):
    """Auto-detect all dimensions from existing result files across all algorithms."""
    dims = set()
    pattern = re.compile(r"_D(\d+)\.txt$")
    for algo in algos:
        algo_dir = os.path.join(results_dir, algo)
        if not os.path.isdir(algo_dir):
            continue
        for func_dir in os.listdir(algo_dir):
            func_path = os.path.join(algo_dir, func_dir)
            if not os.path.isdir(func_path):
                continue
            for fname in os.listdir(func_path):
                # Skip solution files
                if "_solution" in fname:
                    continue
                m = pattern.search(fname)
                if m:
                    dims.add(int(m.group(1)))
    return sorted(dims)


def _add_wilcoxon_test(results_dir, algos):
    """
    Compute pairwise Wilcoxon rank-sum tests between all algorithm pairs.
    Write results to: results/statistical_comparison.csv
    """
    stat_output = os.path.join(results_dir, "statistical_comparison.csv")

    with open(stat_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algo_A", "Algo_B", "p_value", "U_statistic",
                          "Significant_05", "Winner"])

        for algo_a, algo_b in itertools.combinations(algos, 2):
            fitness_a = []
            fitness_b = []

            # Collect all mean fitness values for both algorithms
            for func_id in range(1, 31):
                if func_id == 2:
                    continue

                dims = [2, 10, 20, 30, 50, 100] if (1 <= func_id <= 10 or 21 <= func_id <= 28) \
                       else [10, 20, 30, 50, 100]

                for dim in dims:
                    path_a = os.path.join(results_dir, algo_a, f"F{func_id}",
                                          f"{algo_a}_F{func_id}_D{dim}.txt")
                    path_b = os.path.join(results_dir, algo_b, f"F{func_id}",
                                          f"{algo_b}_F{func_id}_D{dim}.txt")

                    stats_a = parse_result_file(path_a)
                    stats_b = parse_result_file(path_b)

                    if stats_a and stats_b:
                        try:
                            fit_a = float(stats_a.get("Mean Fitness", "nan"))
                            fit_b = float(stats_b.get("Mean Fitness", "nan"))
                            if not np.isnan(fit_a) and not np.isnan(fit_b):
                                fitness_a.append(fit_a)
                                fitness_b.append(fit_b)
                        except (ValueError, TypeError):
                            pass

            if len(fitness_a) > 1 and len(fitness_b) > 1:
                u_stat, p_value = ranksums(fitness_a, fitness_b)
                sig = "Yes" if p_value < 0.05 else "No"

                # Winner is algo with lower mean fitness (closer to ideal)
                mean_a = np.mean(fitness_a)
                mean_b = np.mean(fitness_b)
                winner = algo_a if mean_a < mean_b else algo_b

                writer.writerow([algo_a, algo_b, f"{p_value:.6e}",
                                 f"{u_stat:.6e}", sig, winner])

    print(f"Statistical comparison saved to: {stat_output}")


def build_summary():
    results_dir = "results"
    output_file = os.path.join(results_dir, "summary.csv")

    # Discover algorithm directories
    algos = _discover_algorithms(results_dir)
    if not algos:
        print("No algorithm directories found in results/. Nothing to summarise.")
        return

    print(f"Detected algorithms: {algos}")

    # Auto-detect dimensions from result files
    dimensions = _detect_dimensions(results_dir, algos)
    if not dimensions:
        print("No result files found in results/ directory.")
        return

    print(f"Detected dimensions: {dimensions}")

    # CSV columns: Algorithm | Function | per-dimension stats
    header = ["Algorithm", "Function"]
    for dim in dimensions:
        prefix = f"D{dim}_"
        header.extend([
            f"{prefix}Ideal",
            f"{prefix}Best Fitness",
            f"{prefix}Mean Fitness",
            f"{prefix}Worst Fitness",
            f"{prefix}Std Dev",
            f"{prefix}SEM",
            f"{prefix}Success Rate",
        ])

    rows = []

    for algo in algos:
        for func_id in range(1, 31):
            if func_id == 2:
                continue  # F2 is deprecated in CEC2017
            row = [algo, f"F{func_id}"]

            for dim in dimensions:
                filepath = os.path.join(
                    results_dir, algo, f"F{func_id}",
                    f"{algo}_F{func_id}_D{dim}.txt"
                )
                stats = parse_result_file(filepath)

                if stats:
                    def _get(key):
                        try:
                            return float(stats.get(key, "—"))
                        except (ValueError, TypeError):
                            return "—"

                    ideal = _get("Ideal")
                    best = _get("Best Fitness")
                    mean = _get("Mean Fitness")
                    worst = _get("Worst Fitness")
                    sd = _get("Std Dev")
                    sem = _get("SEM")
                    success = stats.get("Success Rate", "—")

                    row.extend([
                        f"{ideal:.6e}" if isinstance(ideal, (int, float)) else ideal,
                        f"{best:.6e}" if isinstance(best, (int, float)) else best,
                        f"{mean:.6e}" if isinstance(mean, (int, float)) else mean,
                        f"{worst:.6e}" if isinstance(worst, (int, float)) else worst,
                        f"{sd:.6e}" if isinstance(sd, (int, float)) else sd,
                        f"{sem:.6e}" if isinstance(sem, (int, float)) else sem,
                        f"{success}%" if success != "—" else "—",
                    ])
                else:
                    # No results for this algo/function/dimension
                    row.extend(["—"] * 7)

            rows.append(row)

    # Write CSV
    os.makedirs(results_dir, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Summary saved to: {output_file}")
    print(f"Algorithms: {algos} | Functions: F1–F30 | Dimensions: {dimensions}")
    print(f"Total entries: {len(rows)}")
    print(f"Columns: {len(header)}")

    # Pairwise Wilcoxon rank-sum tests
    if len(algos) >= 2:
        print("\nPerforming pairwise Wilcoxon rank-sum tests...")
        _add_wilcoxon_test(results_dir, algos)


if __name__ == "__main__":
    build_summary()
