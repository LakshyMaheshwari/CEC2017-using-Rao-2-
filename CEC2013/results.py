import os
import numpy as np

from CEC2013.config import MAX_ITERATIONS


def save_results(func_id, dimension, stats, total_time, run_times,
                 best_solution, best_solutions, runs, max_fes,
                 fes_used_list, algo_name=None):
    """
    Save results in a simplified fitness-based format.

    stats: dict with Best Fitness, Mean Fitness, Worst Fitness, Std Dev, SEM, etc.
    best_solution: best solution vector found
    best_solutions: list of best solutions for each run
    runs: number of runs
    max_fes: maximum function evaluations per run
    fes_used_list: list of FES used per run
    """
    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_F{func_id}"
    else:
        folder = f"results/F{func_id}"
        prefix = f"F{func_id}"
    os.makedirs(folder, exist_ok=True)

    file_path = f"{folder}/{prefix}_D{dimension}.txt"

    with open(file_path, "w") as f:
        # ── Summary statistics (fitness-based) ──
        f.write(f"Best Fitness\t{stats['Best Fitness']:.6e}\n")
        f.write(f"Mean Fitness\t{stats['Mean Fitness']:.6e}\n")
        f.write(f"Worst Fitness\t{stats['Worst Fitness']:.6e}\n")
        f.write(f"Std Dev\t{stats['Std Dev']:.6e}\n")
        f.write(f"SEM\t{stats['SEM']:.6e}\n")
        f.write(f"Ideal\t{stats['Ideal']:.6e}\n")
        f.write(f"Runs\t{runs}\n")
        f.write(f"Max FES\t{max_fes}\n")
        f.write(f"Iterations\t{MAX_ITERATIONS}\n")
        f.write(f"Total Time\t{total_time:.2f}\n")
        f.write(f"Avg Time\t{np.mean(run_times):.2f}\n")
        f.write(f"Success Rate\t{stats['Success Rate']:.1f}\n")

        # ── Per-run FES usage ──
        f.write(f"\nPer-run FES used:\n")
        for i, fes in enumerate(fes_used_list):
            f.write(f"Run {i+1}\t{fes}\n")

    print(f"Results saved: {file_path}")

    # Save best solution to a separate file
    if best_solution is not None:
        solution_path = f"{folder}/{prefix}_D{dimension}_solution.txt"
        with open(solution_path, "w") as f:
            f.write(f"Best solution for F{func_id} D{dimension}:\n")
            f.write("Decision variables:\n")
            for i, x in enumerate(best_solution):
                f.write(f"x{i+1}\t{x:.6e}\n")
        print(f"Best solution saved: {solution_path}")
