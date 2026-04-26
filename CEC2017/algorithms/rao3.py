import random
from typing import Tuple, List
import numpy as np

from ..utils.population import initialize_population
from ..utils.bounds import apply_bounds
from ..functions.core import evaluate, get_fes


def rao3(
    pop_size: int,
    D: int,
    lb: float,
    ub: float,
    max_fes: int,
    func_id: int,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    Rao-3 (Rao algorithm 3) metaheuristic optimization algorithm.

    This implementation adheres strictly to Function Evaluation (FES) 
    termination criteria to comply with official benchmark constraints.
    Fitness values are cached locally to ensure a 1:1 FES-to-evaluation ratio 
    and prevent computational redundancy.

    Rao-3 extends Rao-2 by adding a third perturbation term based on the
    interaction between the current solution and a second randomly selected
    partner solution, in addition to the best-worst directional term and
    the first partner interaction term from Rao-2.

    Update equation:
        x_new = x + r1*(best - worst) + r2*(|xi - xk| or |xk - xi|)
                  + r3*(|xi - xl| or |xl - xi|)
    where xk and xl are two distinct randomly selected partner solutions.

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

    # ── Initialization Phase: Compute fitness of the initial population ──
    # Note: Evaluates all `pop_size` candidates, consuming `pop_size` FES.
    fitness = np.array([evaluate(population[i], func_id)[0]
                        for i in range(pop_size)])

    running_best_f = np.min(fitness)
    running_best   = population[np.argmin(fitness)].copy()

    fitness_history = [(get_fes(), running_best_f)]

    # ── Main Optimization Loop ──
    while get_fes() < max_fes:

        # ── Global Optimization State Extraction ──
        # Extract the globally best and worst performing solutions from the 
        # current generation. Accomplished via cache lookup to conserve FES.
        best_idx  = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))
        best  = population[best_idx]
        worst = population[worst_idx]

        for i in range(pop_size):

            # Enforce strict FES budget constraint mid-generation validation
            if get_fes() >= max_fes:
                break

            x = population[i]

            # ── First Candidate Partner Selection (xk) ──
            # Stochastically select the first interacting candidate solution 
            # ensuring xk is distinct from the target solution x.
            k = random.randint(0, pop_size - 1)
            while k == i:
                k = random.randint(0, pop_size - 1)
            xk = population[k]

            # ── Second Candidate Partner Selection (xl) ──
            # Stochastically select the second interacting candidate solution 
            # ensuring xl is distinct from both x and xk.
            l = random.randint(0, pop_size - 1)
            while l == i or l == k:
                l = random.randint(0, pop_size - 1)
            xl = population[l]

            # ── Metaphor-less Perturbation Equation (Rao-3) ──
            r1    = np.random.random(D)
            r2    = np.random.random(D)
            r3    = np.random.random(D)
            coin1 = np.random.random(D)
            coin2 = np.random.random(D)

            # First interaction term: directional relationship between
            # candidate x and its first partner xk (same as Rao-2's partner term)
            second_term = np.where(
                coin1 < 0.5,
                np.abs(x) - np.abs(xk),
                np.abs(xk) - np.abs(x)
            )

            # Second interaction term: directional relationship between
            # candidate x and its second partner xl (Rao-3 extension)
            third_term = np.where(
                coin2 < 0.5,
                np.abs(x) - np.abs(xl),
                np.abs(xl) - np.abs(x)
            )

            # Synthesize the new candidate solution
            x_new = x + r1 * (best - worst) + r2 * second_term + r3 * third_term
            
            # Apply boundary constraints to prevent out-of-bounds exploration
            x_new = apply_bounds(x_new, lb, ub)

            # ── Candidate Evaluation ──
            # Probe the true landscape fitness of the perturbated candidate.
            f_new, _ = evaluate(x_new, func_id)

            # ── Elitist Selection Mechanism ──
            # Update population registry solely if the perturbated vector 
            # achieves a strictly dominant or equal objective fitness value.
            if f_new <= fitness[i]:
                population[i] = x_new
                fitness[i]    = f_new

            # ── Global Best Tracking ──
            if fitness[i] < running_best_f:
                running_best_f = fitness[i]
                running_best   = population[i].copy()
                fitness_history.append((get_fes(), running_best_f))

    return running_best, fitness_history