"""
Automated benchmark runner — runs all 30 CEC2017 functions
across all configured dimensions for one or more algorithms.

Usage:
    python -m CEC2017.run_all
"""



from CEC2017.algorithms import ALGORITHMS
from CEC2017.summarize import build_summary
from CEC2017.config import POP_SIZE, MAX_FES_FACTOR, RUNS, LOWER_BOUND, UPPER_BOUND
from CEC2017.runner import run_experiment


def main():
    """
    Automated script to run all 30 CEC2017 functions
    across all configured dimensions for every algorithm.
    """
    algo_names = list(ALGORITHMS.keys())

    for algo_name in algo_names:
        print(f"\n{'#'*60}")
        print(f" ALGORITHM: {algo_name.upper()}")
        print(f"{'#'*60}")

        for func_id in range(1, 31):
            # F2 (Shifted and Rotated Schwefel's Function) is officially deprecated
            # in the CEC2017 technical report. It was removed after the competition
            # due to implementation inconsistencies across platforms. Skip it.
            if func_id == 2:
                continue

            print(f"\n{'='*60}")
            print(f" {algo_name.upper()} — FUNCTION F{func_id}")
            print(f"{'='*60}")

            # CEC2017 official dimensions per function group:
            # F1–F10, F21–F28 support D=2 as well as the standard set.
            if 1 <= func_id <= 10 or 21 <= func_id <= 28:
                dims_to_run = [2, 10, 20, 30, 50, 100]
            else:
                dims_to_run = [10, 20, 30, 50, 100]

            for dim in dims_to_run:
                max_fes = MAX_FES_FACTOR * dim
                print(f"\n[RUNNING] {algo_name.upper()} F{func_id} | D={dim} | MaxFES={max_fes} | Runs={RUNS}")

                try:
                    run_experiment(
                        algo_name,
                        func_id,
                        dim,
                        LOWER_BOUND,
                        UPPER_BOUND,
                        POP_SIZE,
                        max_fes,
                        RUNS,
                    )
                except Exception as e:
                    print(f"ERROR in {algo_name.upper()} F{func_id} D{dim}: {e}")
                    continue

    print("\n\n" + "#"*60)
    print(" ALL ALGORITHMS × ALL FUNCTIONS COMPLETED")
    print("#"*60 + "\n")

    # Automatically generate the summary CSV
    print("Generating final summary CSV...")
    build_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C). Exiting cleanly.")
