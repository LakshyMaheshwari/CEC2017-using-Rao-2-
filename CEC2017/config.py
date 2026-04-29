DIMENSIONS = [2, 10, 20, 30, 50, 100]

POP_SIZE = 50                          # Population size
MAX_FES = 500_000                      # FE = Iteration × POP_SIZE = 10,000 × 50
MAX_ITERATIONS = MAX_FES // POP_SIZE   # Iteration column in CSV = 10,000 per run
RUNS = 10                              # 10 runs × 10,000 iter = 1,00,000 Total Iterations

LOWER_BOUND = -100
UPPER_BOUND = 100
