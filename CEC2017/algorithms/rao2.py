import random
import numpy as np

from utils.population import initialize_population
from utils.bounds import apply_bounds
from functions.core import evaluate, get_fes, reset_fes


def rao2(pop_size, D, lb, ub, max_fes, func_id):
    """
    Rao-2 (Rao algorithm 2) metaheuristic optimization algorithm.
    
    This implementation adheres strictly to Function Evaluation (FES) 
    termination criteria to comply with official benchmark constraints.
    Fitness values are cached locally to ensure a 1:1 FES-to-evaluation ratio 
    and prevent computational redundancy.
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

            # ── Candidate Partner Selection ──
            # Stochastically select an interacting candidate solution (xl) 
            # ensuring xl is distinct from the target sequence x.
            l = random.randint(0, pop_size - 1)
            while l == i:
                l = random.randint(0, pop_size - 1)
            xl = population[l]

            # ── Metaphor-less Perturbation Equation (Rao-2) ──
            r1   = np.random.random(D)
            r2   = np.random.random(D)
            coin = np.random.random(D)

            # Determine the directional vector relationship between candidate x 
            # and its partner xl, executing individual dimension-wise selections.
            second_term = np.where(
                coin < 0.5,
                np.abs(x) - np.abs(xl),
                np.abs(xl) - np.abs(x)
            )

            # Synthesize the new candidate solution
            x_new = x + r1 * (best - worst) + r2 * second_term
            
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