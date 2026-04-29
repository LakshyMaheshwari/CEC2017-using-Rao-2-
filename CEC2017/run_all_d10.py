"""
Automated benchmark runner — D=10 ONLY.
Runs all 30 CEC2017 functions at dimension 10.

Usage:
    python -m CEC2017.run_all_d10
"""

import os
import traceback

from CEC2017.algorithms import ALGORITHMS
from CEC2017.summarize import build_summary
from CEC2017.config import POP_SIZE, MAX_FES, RUNS, LOWER_BOUND, UPPER_BOUND
from CEC2017.runner import run_experiment, write_comparison_csv


def _prompt_algorithm():
    """Prompt user to pick one or all algorithms."""
    algo_list = list(ALGORITHMS.keys())

    while True:
        print("=" * 60)
        print("  CEC2017 — Run All Functions (D=10 ONLY)")
        print("=" * 60)
        print("  Select algorithm to benchmark:")
        for i, algo in enumerate(algo_list, 1):
            print(f"    {i}. {algo.upper()}")
        print(f"    {len(algo_list) + 1}. ALL algorithms")

        choice = input(f"  Enter choice [1-{len(algo_list) + 1}]: ").strip()

        try:
            choice_num = int(choice)
        except ValueError:
            print(f"  ✗ Invalid input. Please enter 1-{len(algo_list) + 1}.\n")
            continue

        if 1 <= choice_num <= len(algo_list):
            return [algo_list[choice_num - 1]]
        elif choice_num == len(algo_list) + 1:
            return algo_list
        else:
            print(f"  ✗ Out of range. Please enter 1-{len(algo_list) + 1}.\n")


def main():
    """Run all 30 CEC2017 functions at D=10 only."""
    algo_names = _prompt_algorithm()

    dim = 10  # Fixed dimension

    for algo_name in algo_names:
        print(f"\n{'#'*60}")
        print(f" ALGORITHM: {algo_name.upper()} | DIMENSION: D={dim}")
        print(f"{'#'*60}")

        for func_id in range(1, 31):
            if func_id == 2:
                continue

            # Skip already-computed results
            result_file = f"results/{algo_name}/F{func_id}/{algo_name}_F{func_id}_D{dim}.txt"
            if os.path.exists(result_file):
                print(f"[SKIP] {algo_name.upper()} F{func_id} D{dim} already computed")
                continue

            print(f"\n[RUNNING] {algo_name.upper()} F{func_id} | D={dim} | MaxFES={MAX_FES} | Runs={RUNS}")

            try:
                run_experiment(
                    algo_name, func_id, dim,
                    LOWER_BOUND, UPPER_BOUND,
                    POP_SIZE, MAX_FES, RUNS,
                )
            except Exception as e:
                print(f"ERROR in {algo_name.upper()} F{func_id} D{dim}:")
                traceback.print_exc()
                continue

    print("\n\n" + "#"*60)
    print(f" ALL FUNCTIONS COMPLETED (D={dim})")
    print("#"*60 + "\n")

    write_comparison_csv()
    print("Comparison summary written to results/comparison_summary.csv")

    print("Generating final summary CSV...")
    build_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C). Exiting cleanly.")
