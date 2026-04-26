"""
Crawl results/ folder and build a single Big Summary CSV.
Extracts Mean Error (and other stats) from every F{id}_D{dim}.txt file.

Usage:
    python summarize.py
"""

import os
import csv
import re
import sys

# Add parent directory to sys.path to allow running as a script from within the package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


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


def parse_best_solution(filepath, dimension):
    """Parse best solution file and return decision variables."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            decision_vars = {}
            for line in lines[1:]:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2 and parts[0].startswith('x'):
                        var_name = parts[0]
                        decision_vars[var_name] = float(parts[1])
            return decision_vars
    except FileNotFoundError:
        return None


def _detect_dimensions(results_dir):
    """Auto-detect all dimensions from existing result files."""
    dims = set()
    pattern = re.compile(r"F\d+_D(\d+)\.txt$")
    for func_dir in os.listdir(results_dir):
        func_path = os.path.join(results_dir, func_dir)
        if not os.path.isdir(func_path):
            continue
        for fname in os.listdir(func_path):
            m = pattern.match(fname)
            if m:
                dims.add(int(m.group(1)))
    return sorted(dims)


def build_summary():
    results_dir = "results"
    output_file = os.path.join(results_dir, "summary.csv")

    # Auto-detect dimensions from result files
    dimensions = _detect_dimensions(results_dir)
    if not dimensions:
        print("No result files found in results/ directory.")
        return

    print(f"Detected dimensions: {dimensions}")

    # CSV columns (per dimension block):
    # Best Value | Best Error | Worst Error | Mean Error | Std Dev | Std Error | x1..xD
    header = ["Function"]

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

    for func_id in range(1, 31):
        row = [f"F{func_id}"]
        func_id * 100

        for dim in dimensions:
            filepath = os.path.join(results_dir, f"F{func_id}", f"F{func_id}_D{dim}.txt")
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
                # No results for this function/dimension
                row.extend(["—"] * 6)

        rows.append(row)

    # Write CSV
    os.makedirs(results_dir, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Summary saved to: {output_file}")
    print(f"Functions: F1–F30 | Dimensions: {dimensions}")
    print(f"Total entries: {len(rows)}")
    print(f"Columns: {len(header)}")


if __name__ == "__main__":
    build_summary()
