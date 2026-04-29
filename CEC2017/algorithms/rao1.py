from typing import Tuple, List
import numpy as np

from ..utils.population import initialize_population
from ..utils.bounds import apply_bounds
from ..functions.core import evaluate, get_fes


def rao1(
    pop_size: int,
    D: int,
    lb: float,
    ub: float,
    max_fes: int,
    func_id: int,
    early_stop_value: float = None,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    Rao-1 (Rao algorithm 1) metaheuristic optimization algorithm.

    This implementation adheres strictly to Function Evaluation (FES) 
    termination criteria to comply with official benchmark constraints.
    Fitness values are cached locally to ensure a 1:1 FES-to-evaluation ratio 
    and prevent computational redundancy.

    Rao-1 uses only the best-worst directional perturbation without the
    random partner interaction term present in Rao-2.

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

        # Early-stop: if ideal value is reached, terminate
        if early_stop_value is not None and running_best_f <= early_stop_value + 1e-8:
            break

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

            # ── Metaphor-less Perturbation Equation (Rao-1) ──
            # Rao-1 uses a single random vector scaled by the difference
            # between the best and worst solutions in the population.
            r1 = np.random.random(D)

            # Synthesize the new candidate solution
            x_new = x + r1 * (best - worst)
            
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