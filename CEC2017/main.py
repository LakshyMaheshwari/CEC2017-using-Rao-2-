import argparse

from CEC2017.runner import run_experiment, write_comparison_csv
from CEC2017.config import POP_SIZE, MAX_FES, RUNS, LOWER_BOUND, UPPER_BOUND
from CEC2017.algorithms import ALGORITHMS


def _prompt_algorithm():
    """Display algorithm menu and return a list of algo_name strings."""
    algo_list = list(ALGORITHMS.keys())

    while True:
        print("=" * 60)
        print("  CEC2017 Benchmark Suite")
        print("=" * 60)
        print("  Select algorithm:")
        for i, algo in enumerate(algo_list, 1):
            print(f"    {i}. {algo.upper()}")
        print(f"    {len(algo_list) + 1}. Run ALL algorithms (sequential)")

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


def _prompt_function():
    """Prompt for function ID and validate."""
    while True:
        raw = input("\n  Select function number [1-30]: ").strip()
        try:
            func_id = int(raw)
        except ValueError:
            print(f"  ✗ '{raw}' is not a valid integer.\n")
            continue
        if 1 <= func_id <= 30:
            return func_id
        print(f"  ✗ Function {func_id} is out of range. Must be 1-30.\n")


def main():
    parser = argparse.ArgumentParser(
        description="CEC2017 Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m CEC2017.main                          # Interactive menu
  python -m CEC2017.main --algo rao2 --func 1     # Run RAO-2 on F1
  python -m CEC2017.main --all                    # Run all algos
  python -m CEC2017.main --runs 10 --max-fes 50000  # Custom params
        """
    )
    parser.add_argument("--algo", choices=list(ALGORITHMS.keys()),
                        help="Algorithm to run")
    parser.add_argument("--func", type=int, choices=range(1, 31),
                        help="Function ID", metavar="FUNC_ID")
    parser.add_argument("--all", action="store_true",
                        help="Run all algorithms")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed results")
    parser.add_argument("--runs", type=int, default=RUNS,
                        help=f"Number of independent runs (default: {RUNS})")
    parser.add_argument("--max-fes", type=int, default=MAX_FES,
                        help=f"Max function evaluations per run (default: {MAX_FES})")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE,
                        help=f"Population size (default: {POP_SIZE})")

    args = parser.parse_args()

    # Determine algorithms to run
    if args.all:
        algo_names = list(ALGORITHMS.keys())
    elif args.algo:
        algo_names = [args.algo]
    else:
        # Interactive mode
        algo_names = _prompt_algorithm()

    # Determine function ID
    if args.func:
        func_id = args.func
    else:
        func_id = _prompt_function()

    # Determine dimension list based on function ID
    if func_id in (29, 30):
        dims_to_run = [2, 10, 20, 30, 50, 100]
    elif 1 <= func_id <= 10 or 21 <= func_id <= 28:
        dims_to_run = [2, 10, 20, 30, 50, 100]
    else:
        dims_to_run = [10, 20, 30, 50, 100]

    try:
        for algo_name in algo_names:
            for dim in dims_to_run:

                # --resume: Skip already-computed results
                if args.resume:
                    import os
                    result_file = f"results/{algo_name}/F{func_id}/{algo_name}_F{func_id}_D{dim}.txt"
                    if os.path.exists(result_file):
                        print(f"[SKIP] {algo_name.upper()} F{func_id} D{dim} already computed")
                        continue

                print(f"\n{'─' * 60}")
                print(f"Running {algo_name.upper()} | F{func_id} | D={dim} | MaxFES={args.max_fes} | Runs={args.runs}")
                print(f"{'─' * 60}")
                run_experiment(
                    algo_name,
                    func_id,
                    dim,
                    LOWER_BOUND,
                    UPPER_BOUND,
                    args.pop_size,
                    args.max_fes,
                    args.runs,
                )
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C). Exiting cleanly.")
        return

    # Write batch comparison CSV at the end
    write_comparison_csv()
    print("Comparison summary written to results/comparison_summary.csv")

    # If user ran ALL algorithms, generate the summary CSV at the end
    if len(algo_names) == len(list(ALGORITHMS.keys())):
        print("\n" + "=" * 60)
        print("  Generating final summary CSV...")
        print("=" * 60)
        from CEC2017.summarize import build_summary
        build_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C). Exiting cleanly.")
