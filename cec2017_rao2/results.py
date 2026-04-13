import numpy as np
import os

from .config import FES_CHECKPOINTS


def save_results(func_id, dimension, error_matrix, stats, total_time):
    """
    Save results in CEC2017 official .txt format.

    error_matrix: numpy array of shape (14, runs) — error values at checkpoints
    stats: dict with Best, Worst, Median, Mean, Std
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
        f.write(f"Best\t{stats['Best']:.6e}\n")
        f.write(f"Worst\t{stats['Worst']:.6e}\n")
        f.write(f"Median\t{stats['Median']:.6e}\n")
        f.write(f"Mean\t{stats['Mean']:.6e}\n")
        f.write(f"Std\t{stats['Std']:.6e}\n")
        f.write(f"Time\t{total_time:.2f}\n")

    print(f"Results saved: {file_path}")