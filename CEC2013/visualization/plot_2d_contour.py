import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


from CEC2013.functions.core import get_fes, fes_counter
from CEC2013.functions.get_function import get_function


def plot_2d_contour(func_id, best_solution, lb, ub, algo_name=None):
    """Plot 2D contour map of a 2D function landscape. D=2 only."""

    # ── Save and restore FES so visualization doesn't corrupt run stats ──
    fes_before = get_fes()

    # 1. Build grid (resolution=100)
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)

    # 2. Evaluate all points — uses the benchmark function directly,
    # bypassing the FES counter to keep visualization free of side effects
    func = get_function(func_id)["objective"]

    # Vectorized evaluation for efficiency
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z_flat = np.array([func(p) for p in points])
    Z = Z_flat.reshape(X.shape)

    # ── Restore FES counter ──
    fes_counter.set(fes_before)

    # 3. Create the Plot
    fig = plt.figure(figsize=(8, 8))  # Square aspect ratio to match the image
    ax = fig.add_subplot(111)

    # Use plt.contour with many levels (50) and a jet colormap
    # to create the dense concentric lines.
    # linewidths=0.8 matches the thin style of the lines in the image.
    ax.contour(X, Y, Z, levels=50, cmap='jet', linewidths=0.8)

    # 4. Add the Best Solution Marker (the small red '+' or '*' in the center)
    # The 'marker' choice '*' or '+' approximates the target.
    # The image has a small, light blue outline around the marker,
    # simulated with 'edgecolors'.
    ax.scatter(best_solution[0], best_solution[1], color='red',
               s=150, edgecolors='lightblue', linewidth=1.5, marker='*', zorder=10)

    # 5. Styling to Match the Image
    ax.set_title(f"F{func_id} 2D Contour Map", fontsize=14)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)

    # Set exact limits and ensure aspect is equal for circles
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.set_aspect('equal')  # Prevents distortion, making circles look like circles

    # 6. Set specific tick marks to match the provided image (-100, -80, ..., 100)
    ticks = np.arange(lb, ub + 20, 20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # 7. Save and Handle the Folder Structure
    # This matches the requested format: results > Fn > Fn_2D.png
    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_F{func_id}"
    else:
        folder = f"results/F{func_id}"
        prefix = f"F{func_id}"
    os.makedirs(folder, exist_ok=True)

    save_path = f"{folder}/{prefix}_2D.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph saved successfully at: {save_path}")
