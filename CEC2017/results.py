import os


def save_results(func_id, dimension, error_matrix, stats, total_time, best_solution, best_solutions, runs):
    """
    Save results in CEC2017 official .txt format with additional statistics.

    error_matrix: numpy array of shape (14, runs) — error values at checkpoints
    stats: dict with Best, Worst, Median, Mean, Std
    best_solution: best solution vector found
    best_solutions: list of best solutions for each run
    runs: number of runs
    """
    folder = f"results/F{func_id}"
    os.makedirs(folder, exist_ok=True)

    file_path = f"{folder}/F{func_id}_D{dimension}.txt"

    with open(file_path, "w") as f:
        # ── 14 × RUNS error matrix ──
        rows, cols = error_matrix.shape
        for i in range(rows):
            row_vals = []
            for j in range(cols):
                row_vals.append(f"{error_matrix[i, j]:.6e}")
            f.write("\t".join(row_vals) + "\n")

        # ── Summary statistics ──
        f.write("\n")
        f.write(f"Best Value\t{stats['Best Value']:.6e}\n")
        f.write(f"Best Error\t{stats['Best Error']:.6e}\n")
        f.write(f"Worst Error\t{stats['Worst Error']:.6e}\n")
        f.write(f"Median Error\t{stats['Median Error']:.6e}\n")
        f.write(f"Mean Error\t{stats['Mean Error']:.6e}\n")
        f.write(f"Std Dev\t{stats['Std Dev']:.6e}\n")
        f.write(f"Std Error\t{stats['Std Error']:.6e}\n")
        f.write(f"Time\t{total_time:.2f}\n")

        # ── Additional statistics for summary generation ──
        f.write(f"Ideal\t{func_id * 100:.6e}\n")
        f.write(f"Runs\t{runs}\n")

    print(f"Results saved: {file_path}")

    # Save best solution to a separate file
    if best_solution is not None:
        solution_path = f"{folder}/F{func_id}_D{dimension}_solution.txt"
        with open(solution_path, "w") as f:
            f.write(f"Best solution for F{func_id} D{dimension}:\n")
            f.write("Decision variables:\n")
            for i, x in enumerate(best_solution):
                f.write(f"x{i+1}\t{x:.6e}\n")
        print(f"Best solution saved: {solution_path}")
