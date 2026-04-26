import numpy as np
import matplotlib.pyplot as plt
import os


def plot_convergence(all_histories, func_id, dimension, f_star, algo_name=None):
    """
    Plot convergence curve: error (Fi(x) - Fi*) vs FES.

    all_histories: list of histories, where each history is a list of
                   (fes_count, best_fitness) tuples from one run.
    f_star: optimal value Fi* for the function.
    """
    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_"
    else:
        folder = f"results/F{func_id}"
        prefix = ""
    os.makedirs(folder, exist_ok=True)

    plt.figure()

    # Find the maximum FES across all runs to create a common x-axis
    max_fes = max(fes for history in all_histories for fes, _ in history)

    # Create a common FES grid and interpolate each run onto it
    common_fes = np.linspace(0, max_fes, 500)

    interpolated = []
    for history in all_histories:
        fes_vals = [fes for fes, _ in history]
        fit_vals = [fit for _, fit in history]
        # Step interpolation (last-value-carried-forward) — NOT linear interp
        idx = np.searchsorted(fes_vals, common_fes, side='right') - 1
        idx = np.clip(idx, 0, len(fit_vals) - 1)
        interp_fit = np.array([fit_vals[i] for i in idx])
        # convert to error
        interp_error = np.maximum(interp_fit - f_star, 0.0)
        interpolated.append(interp_error)

    avg_error = np.mean(interpolated, axis=0)

    plt.plot(common_fes, avg_error)
    plt.xlabel("Function Evaluations (FES)")
    plt.ylabel("Error  F(x) − F*")
    plt.title(f"F{func_id} Convergence (D={dimension})")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    file_path = f"{folder}/{prefix}convergence_D{dimension}.png"
    plt.savefig(file_path)
    plt.close()
