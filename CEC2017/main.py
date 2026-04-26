from runner import run_experiment
from config import POP_SIZE, MAX_FES_FACTOR, RUNS, LOWER_BOUND, UPPER_BOUND


def main():
    print("Select Function (1–30): ")
    func_id = int(input())

    if func_id < 1 or func_id > 30:
        print("Invalid function ID")
        return

    if func_id == 29 or func_id == 30:
        dims_to_run = [10, 20, 30, 50, 100]
    elif 1 <= func_id <= 10 or 21 <= func_id <= 28:
        dims_to_run = [2, 10, 20, 30, 50, 100]
    else:
        dims_to_run = [10, 20, 30, 50, 100]

    for dim in dims_to_run:
        max_fes = MAX_FES_FACTOR * dim
        print(f"\nRunning F{func_id} | Dimension = {dim} | MaxFES = {max_fes}")
        run_experiment(
            func_id,
            dim,
            LOWER_BOUND,
            UPPER_BOUND,
            POP_SIZE,
            max_fes,
            RUNS
        )


if __name__ == "__main__":
    main()
