import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from CEC2013.functions.core import get_fes, fes_counter
from CEC2013.functions.get_function import get_function


def plot_3d_surface(func_id, best_solution, lb, ub, resolution=100, algo_name=None):
    """
    Plot 3D surface of a 2D function landscape.
    Vectorized — evaluates all grid points in one pass, no nested loop.
    FES counter is preserved — visualization does not corrupt run statistics.
    """

    # ── Save and restore FES so visualization doesn't corrupt run stats ──
    fes_before = get_fes()

    # ── Build grid ──
    x = np.linspace(lb, ub, resolution)
    y = np.linspace(lb, ub, resolution)
    X, Y = np.meshgrid(x, y)

    # ── Vectorized evaluation ──
    # Stack all (x, y) pairs into shape (resolution*resolution, 2)
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate all points — uses the benchmark function directly,
    # bypassing the FES counter to keep visualization free of side effects
    func = get_function(func_id)["objective"]

    Z_flat = np.array([func(points[i]) for i in range(len(points))])
    Z = Z_flat.reshape(X.shape)

    # ── Best solution height ──
    z_best = func(best_solution)

    # ── Restore FES counter ──
    # Visualization should never count toward the algorithm budget
    fes_counter.set(fes_before)

    # ── Plot ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        cmap='jet',
        edgecolor='black',
        linewidth=0.1,
        alpha=0.85
    )

    # Floor projection contour
    offset_val = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.05
    ax.contour(X, Y, Z, zdir='z', offset=offset_val, cmap='jet', alpha=0.5)

    # Best solution marker
    ax.scatter(
        best_solution[0], best_solution[1], z_best,
        color='red', s=100, edgecolors='white',
        marker='*', label='Global Best', zorder=10
    )

    ax.view_init(elev=25, azim=-45)
    ax.set_title(f"F{func_id} Optimization Landscape", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Fitness", fontsize=12)
    ax.set_zlim(offset_val, np.max(Z))
    ax.legend()

    if algo_name:
        folder = f"results/{algo_name}/F{func_id}"
        prefix = f"{algo_name}_F{func_id}"
    else:
        folder = f"results/F{func_id}"
        prefix = f"F{func_id}"
    save_path = f"{folder}/{prefix}_3D.png"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"3D surface saved: {save_path}")
