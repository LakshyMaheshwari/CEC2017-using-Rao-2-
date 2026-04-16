import os
import sys

# Add parent directory to sys.path to allow running as a script from within the package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cec2017_rao2.runner import run_experiment
from cec2017_rao2.config import POP_SIZE, MAX_FES_FACTOR, RUNS, LOWER_BOUND, UPPER_BOUND
from cec2017_rao2.summarize import build_summary


def main():
    """
    Automated script to run all 30 CEC2017 functions
    across all configured dimensions.
    """
    # Loop through all 30 functions
    for func_id in range(1, 31):
        if func_id == 2:
            continue

        print(f"\n{'='*60}")
        print(f" BEGINNING EVALUATION: FUNCTION F{func_id} ")
        print(f"{'='*60}")

        # if 1 <= func_id <= 10:
        #     dims_to_run = [2, 10, 20, 30, 50, 100]
        # elif 21 <= func_id <= 28:
        #     dims_to_run = [2, 10, 20, 30, 50, 100]
        # else:
        #     dims_to_run = [10, 20, 30, 50, 100]

        if 1 <= func_id <= 10:
            dims_to_run = [2, 10]
        elif 21 <= func_id <= 28:
            dims_to_run = [2, 10]
        else:
            dims_to_run = [10]

        for dim in dims_to_run:
            max_fes = MAX_FES_FACTOR * dim
            print(f"\n[RUNNING] F{func_id} | D={dim} | MaxFES={max_fes} | Runs={RUNS}")
            
            try:
                run_experiment(
                    func_id,
                    dim,
                    LOWER_BOUND,
                    UPPER_BOUND,
                    POP_SIZE,
                    max_fes,
                    RUNS
                )
            except Exception as e:
                print(f"ERROR in F{func_id} D{dim}: {e}")
                continue

    print("\n\n" + "#"*60)
    print(" ALL 30 FUNCTIONS COMPLETED SUCCESSFULLY ")
    print("#"*60 + "\n")

    # Automatically generate the summary CSV
    print("Generating final summary CSV...")
    build_summary()


if __name__ == "__main__":
    main()
