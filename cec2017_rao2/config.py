DIMENSIONS = [2, 10, 20, 30]

ALGORITHMS = ["rao1", "rao2", "rao3", "fisa"]

POP_SIZE = 30
MAX_FES_FACTOR = 10000    # max_fes = MAX_FES_FACTOR * dimension  (CEC2017 standard)
RUNS = 51                 # CEC2017 official: 51 independent runs

LOWER_BOUND = -100
UPPER_BOUND = 100

# 14 official CEC2017 checkpoints (fractions of MaxFES)
FES_CHECKPOINTS = [
    0.01, 0.02, 0.03, 0.05,
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
]