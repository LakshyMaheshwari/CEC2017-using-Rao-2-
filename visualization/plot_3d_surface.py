import numpy as np
import matplotlib.pyplot as plt
import os

from functions.core import evaluate

def plot_3d_surface(func_id, best_solution, lb, ub):
    # 1. Higher resolution (100) makes the peaks look smooth like the reference
    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 2. Compute function values using your existing evaluate logic
    for i in range(len(X)):
        for j in range(len(X[0])):
            val, _ = evaluate([X[i][j], Y[i][j]], func_id)
            Z[i][j] = val

    # Best solution point height
    z_best, _ = evaluate(best_solution, func_id)

    # 3. Plot Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # The Surface - 'jet' colormap and 'alpha' for that professional look
    # Edgecolor 'black' with very thin width mimics the mesh look in your image
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='black', linewidth=0.1, alpha=0.85)

    # THE "FLOOR" PROJECTION - This is the contour on the bottom
    offset_val = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.05
    ax.contour(X, Y, Z, zdir='z', offset=offset_val, cmap='jet', alpha=0.5)

    # 4. Mark the Best Solution
    ax.scatter(best_solution[0], best_solution[1], z_best, color='red', s=100, edgecolors='white', marker='*', label='Global Best')

    # 5. Styling and View Angle
    ax.view_init(elev=25, azim=-45) # Matches the tilt of your image
    ax.set_title(f"F{func_id} Optimization Landscape", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Fitness", fontsize=12)
    
    # Ensure the floor is visible by setting Z limits
    ax.set_zlim(offset_val, np.max(Z))

    # 6. Save and Display
    folder = f"results/F{func_id}"
    os.makedirs(folder, exist_ok=True)
    
    save_path = f"{folder}/F{func_id}_3D.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph saved successfully at: {save_path}")
