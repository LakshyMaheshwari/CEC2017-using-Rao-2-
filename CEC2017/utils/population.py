import numpy as np


def generate_solution(D, lb, ub):
    return np.random.uniform(lb, ub, size=D)


def initialize_population(pop_size, D, lb, ub):
    return np.random.uniform(lb, ub, size=(pop_size, D))
