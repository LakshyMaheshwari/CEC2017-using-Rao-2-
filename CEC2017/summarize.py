"""
Crawl results/ folder and build a single Big Summary CSV.
Extracts Mean Error (and other stats) from every result .txt file.

Supports the multi-algorithm directory structure:
    results/{algo}/F{id}/{algo}_F{id}_D{dim}.txt

Usage:
    python -m CEC2017.summarize
"""

import os
import csv
import re


# Keys we recognise in result files (order matters for CSV columns)
STAT_KEYS = {
    "Best Value", "Best Error", "Worst Error", "Median Error",
    "Mean Error", "Std Dev", "Std Error", "Time", "Ideal", "Runs",
    # Legacy keys (from old output format) mapped during parsing
    "Best", "Worst", "Median", "Mean", "Std", "StdError",
}


def parse_result_file(filepath):
    """Parse a single result .txt file, return dict of summary stats."""
    stats = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2 and parts[0] in STAT_KEYS:
                    key = parts[0]
                    # Normalise legacy keys to current names
                    if key == "Best":
                        key = "Best Error"
                    elif key == "Worst":
                        key = "Worst Error"
                    elif key == "Median":
                        key = "Median Error"
                    elif key == "Mean":
                        key = "Mean Error"
                    elif key == "Std":
                        key = "Std Dev"
                    elif key == "StdError":
                        key = "Std Error"
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
            # Subdirectories that don't start with F, '.', or '_' are algorithm dirs
            # (e.g. rao1, rao2, rao3, fisa)
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
            f"{prefix}Best Value",
            f"{prefix}Best Error",
            f"{prefix}Worst Error",
            f"{prefix}Mean Error",
            f"{prefix}Std Dev",
            f"{prefix}Std Error",
        ])

    rows = []

    for algo in algos:
        for func_id in range(1, 31):
            row = [algo, f"F{func_id}"]

            for dim in dimensions:
                # New path structure: results/{algo}/F{id}/{algo}_F{id}_D{dim}.txt
                filepath = os.path.join(
                    results_dir, algo, f"F{func_id}",
                    f"{algo}_F{func_id}_D{dim}.txt"
                )
                stats = parse_result_file(filepath)

                if stats:
                    try:
                        best_value = float(stats.get("Best Value", "—"))
                    except (ValueError, TypeError):
                        best_value = "—"

                    try:
                        best_error = float(stats.get("Best Error", "—"))
                    except (ValueError, TypeError):
                        best_error = "—"

                    try:
                        worst_error = float(stats.get("Worst Error", "—"))
                    except (ValueError, TypeError):
                        worst_error = "—"

                    try:
                        mean_error = float(stats.get("Mean Error", "—"))
                    except (ValueError, TypeError):
                        mean_error = "—"

                    try:
                        std_dev = float(stats.get("Std Dev", "—"))
                    except (ValueError, TypeError):
                        std_dev = "—"

                    try:
                        std_error = float(stats.get("Std Error", "—"))
                    except (ValueError, TypeError):
                        std_error = "—"

                    row.extend([
                        f"{best_value:.6e}" if isinstance(best_value, (int, float)) else best_value,
                        f"{best_error:.6e}" if isinstance(best_error, (int, float)) else best_error,
                        f"{worst_error:.6e}" if isinstance(worst_error, (int, float)) else worst_error,
                        f"{mean_error:.6e}" if isinstance(mean_error, (int, float)) else mean_error,
                        f"{std_dev:.6e}" if isinstance(std_dev, (int, float)) else std_dev,
                        f"{std_error:.6e}" if isinstance(std_error, (int, float)) else std_error,
                    ])
                else:
                    # No results for this algo/function/dimension
                    row.extend(["—"] * 6)

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


if __name__ == "__main__":
    build_summary()
