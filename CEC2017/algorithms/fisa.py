import random
from typing import Tuple, List
import numpy as np

from ..utils.population import initialize_population
from ..utils.bounds import apply_bounds
from ..functions.core import evaluate, get_fes


def fisa(
    pop_size: int,
    D: int,
    lb: float,
    ub: float,
    max_fes: int,
    func_id: int,
    early_stop_value: float = None,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    FISA — Fitness-based Individual-Step Algorithm.

    Update equation (three components):
      x_new = x
            + rank[i] * r1 * (best - x)          # exploitation: fitness-scaled pull to global best
            + r2 * sign(fi - fj) * (xj - x)      # peer interaction: attract/repel by fitness comparison
            + r3 * (best - worst)                 # global guidance: align with improvement direction

    Where:
      rank[i] in [0,1]  — normalised fitness rank (0 = best, 1 = worst)
                          Weaker agents take larger exploratory steps toward best.
      sign(fi - fj):      +1 → current is worse than partner  (attract toward xj)
                          -1 → current is better than partner (repel from  xj)
      r1, r2, r3:         independent uniform random vectors in [0,1]^D

    Parameters
    ----------
    pop_size : int
        Population size.
    D : int
        Problem dimension.
    lb : float
        Lower bound.
    ub : float
        Upper bound.
    max_fes : int
        Maximum function evaluations.
    func_id : int
        CEC2017 function ID (1–30, not 2).

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, float]]]
        (best_solution, fitness_history) where:
        - best_solution: shape (D,)
        - fitness_history: list of (fes_count, best_fitness) tuples
    """

    population = initialize_population(pop_size, D, lb, ub)

    # ── Initialization: evaluate the starting population ──
    fitness = np.array([evaluate(population[i], func_id)[0]
                        for i in range(pop_size)])

    running_best_f = np.min(fitness)
    running_best   = population[np.argmin(fitness)].copy()

    fitness_history = [(get_fes(), running_best_f)]

    # ── Main Optimisation Loop ──
    while get_fes() < max_fes:

        # Early-stop: if ideal value is reached, terminate
        if early_stop_value is not None and running_best_f <= early_stop_value + 1e-8:
            break

        best_idx  = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))
        best_f    = fitness[best_idx]
        worst_f   = fitness[worst_idx]
        best_x    = population[best_idx]
        worst_x   = population[worst_idx]

        # Normalised fitness rank for each agent.
        # rank[i] = 0  → best agent  (small exploitation step)
        # rank[i] = 1  → worst agent (large exploratory step)
        f_range = (worst_f - best_f) if worst_f != best_f else 1.0
        rank = (fitness - best_f) / f_range          # shape: (pop_size,)

        for i in range(pop_size):

            # Honour strict FES budget mid-generation
            if get_fes() >= max_fes:
                break

            x  = population[i]
            fi = fitness[i]

            # ── Partner selection (distinct from i) ──
            j = random.randint(0, pop_size - 1)
            while j == i:
                j = random.randint(0, pop_size - 1)
            xj = population[j]
            fj = fitness[j]

            r1 = np.random.random(D)
            r2 = np.random.random(D)
            r3 = np.random.random(D)

            # Component 1 — Fitness-scaled attraction to global best.
            # Weaker agents (rank closer to 1) take larger steps.
            comp1 = rank[i] * r1 * (best_x - x)

            # Component 2 — Fitness-aware peer interaction.
            # If current agent is worse than partner → attract toward partner.
            # If current agent is better than partner → repel from partner.
            if fi > fj:
                comp2 = r2 * (xj - x)   # attract: move toward the better peer
            else:
                comp2 = r2 * (x - xj)   # repel:   move away from the worse peer

            # Component 3 — Global directional bias (best → worst direction).
            # Provides a population-wide improvement signal (analogous to Rao-1).
            comp3 = r3 * (best_x - worst_x)

            x_new = x + comp1 + comp2 + comp3

            # Clip to feasible region
            x_new = apply_bounds(x_new, lb, ub)

            # ── Evaluate and apply greedy selection ──
            f_new, _ = evaluate(x_new, func_id)

            if f_new <= fitness[i]:
                population[i] = x_new
                fitness[i]    = f_new

            # ── Track global best ──
            if fitness[i] < running_best_f:
                running_best_f = fitness[i]
                running_best   = population[i].copy()
                fitness_history.append((get_fes(), running_best_f))

    return running_best, fitness_history