"""
CEC 2013 Test Function Suite - Python translation

Original C code by Jane Jing Liang (2013)
Python translation with exact behavioural match.

Usage:
    import cec13
    f = cec13.cec13_func(x, func_num)
    where x is a 1D array (single point) or 2D array shape (D, pop_size),
    func_num is 1..28.
The folder 'input_data' containing M_D{}.txt and shift_data.txt must exist.
"""

import numpy as np
import math
import os

# Constants
INF = 1e99
EPS = 1e-14
E   = math.e
PI  = math.pi

# ----------------------------------------------------------------------
# Helper transformations
# ----------------------------------------------------------------------
def shiftfunc(x, Os):
    return x - Os

def rotatefunc(x, Mr):
    return Mr @ x

def asyfunc(x, beta):
    nx = len(x)
    y = np.copy(x)
    for i in range(nx):
        if x[i] > 0:
            exponent = 1.0 + beta * i / (nx - 1) * math.sqrt(x[i])
            y[i] = math.pow(x[i], exponent)
    return y

def oszfunc(x):
    nx = len(x)
    y = np.copy(x)
    for i in range(nx):
        if i == 0 or i == nx - 1:
            if x[i] == 0.0:
                y[i] = 0.0
                continue
            xx = math.log(abs(x[i]))
            if x[i] > 0:
                c1, c2 = 10.0, 7.9
            else:
                c1, c2 = 5.5, 3.1
            sx = 1 if x[i] > 0 else -1
            y[i] = sx * math.exp(xx + 0.049 * (math.sin(c1 * xx) + math.sin(c2 * xx)))
    return y

def cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    """Composition function weighting."""
    w = np.zeros(cf_num)
    w_max = 0.0
    for i in range(cf_num):
        diff = x - Os[i]
        sum_sq = np.sum(diff * diff)
        if sum_sq != 0:
            w[i] = math.pow(1.0 / sum_sq, 0.5) * math.exp(-sum_sq / (2.0 * nx * delta[i]**2))
        else:
            w[i] = INF
        if w[i] > w_max:
            w_max = w[i]

    w_sum = np.sum(w)
    if w_max == 0:
        w = np.ones(cf_num)
        w_sum = cf_num

    f_val = 0.0
    for i in range(cf_num):
        f_val += (w[i] / w_sum) * (fit[i] + bias[i])
    return f_val

# ----------------------------------------------------------------------
# Data cache (M and OShift for each dimension)
# ----------------------------------------------------------------------
# Data directory: 'data' folder next to this file
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class _CEC13Data:
    _cache = {}  # nx -> (M, OShift)

    @staticmethod
    def load(nx, data_dir=None):
        if data_dir is None:
            data_dir = DATA_DIR
        if nx in _CEC13Data._cache:
            return _CEC13Data._cache[nx]

        cf_num = 10
        M_path = os.path.join(data_dir, f'M_D{nx}.txt')
        M_flat = np.loadtxt(M_path)
        M = M_flat.reshape(cf_num, nx, nx)

        shift_path = os.path.join(data_dir, 'shift_data.txt')
        O_flat = np.loadtxt(shift_path)
        OShift = O_flat.reshape(cf_num, -1)
        OShift = OShift[:, :nx]  # truncate to dimension

        _CEC13Data._cache[nx] = (M, OShift)
        return M, OShift

# ----------------------------------------------------------------------
# Basic functions (each receives its dedicated rotation matrix Mr, 
# and the next matrix Mr2=None if needed)
# ----------------------------------------------------------------------
def sphere_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr)
    return np.sum(y ** 2)

def ellips_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr)
    y = oszfunc(y)
    cond = np.power(10.0, 6.0 * np.arange(len(x)) / (len(x) - 1))
    return np.sum(cond * y * y)

def bent_cigar_func(x, Os, Mr1, Mr2):
    beta = 0.5
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr1)
    y = asyfunc(y, beta)
    y = rotatefunc(y, Mr2)
    f = y[0]**2
    f += 10.0**6 * np.sum(y[1:] ** 2)
    return f

def discus_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr)
    y = oszfunc(y)
    f = 10.0**6 * y[0]**2 + np.sum(y[1:] ** 2)
    return f

def dif_powers_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr)
    nx = len(y)
    exponents = 2.0 + 4.0 * np.arange(nx) / (nx - 1)
    f = np.sum(np.abs(y) ** exponents)
    return math.sqrt(f)

def rosenbrock_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = y * (2.048 / 100.0)
    y = rotatefunc(y, Mr)
    y = y + 1.0
    f = 0.0
    for i in range(len(y) - 1):
        tmp1 = y[i]**2 - y[i+1]
        tmp2 = y[i] - 1.0
        f += 100.0 * tmp1**2 + tmp2**2
    return f

def schaffer_F7_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr1)
    y = asyfunc(y, 0.5)
    cond = np.power(10.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr2)
    z = np.sqrt(y[:-1]**2 + y[1:]**2)
    tmp = np.sin(50.0 * z**0.2)
    f = np.sum(z**0.5 + z**0.5 * tmp**2)
    return (f / (len(y) - 1))**2

def ackley_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr1)
    y = asyfunc(y, 0.5)
    cond = np.power(10.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr2)
    n = len(y)
    sum1 = np.sum(y**2)
    sum2 = np.sum(np.cos(2.0 * PI * y))
    sum1 = -0.2 * math.sqrt(sum1 / n)
    sum2 /= n
    return E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0

def weierstrass_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = y * (0.5 / 100.0)
    y = rotatefunc(y, Mr1)
    y = asyfunc(y, 0.5)
    cond = np.power(10.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr2)

    a = 0.5
    b = 3.0
    k_max = 20
    f = 0.0
    sum2 = 0.0
    for j in range(k_max + 1):
        sum2 += (a**j) * math.cos(2.0 * PI * (b**j) * 0.5)
    sum2 *= len(y)
    for i in range(len(y)):
        s = 0.0
        for j in range(k_max + 1):
            s += (a**j) * math.cos(2.0 * PI * (b**j) * (y[i] + 0.5))
        f += s
    return f - sum2

def griewank_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = y * (600.0 / 100.0)
    y = rotatefunc(y, Mr1)
    cond = np.power(100.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    # Griewank uses only one rotation (Mr2 unused, kept for interface uniformity)
    s = np.sum(y**2)
    p = np.prod(np.cos(y / np.sqrt(np.arange(1, len(y)+1))))
    return 1.0 + s/4000.0 - p

def rastrigin_func(x, Os, Mr1, Mr2):
    alpha = 10.0
    beta = 0.2
    y = shiftfunc(x, Os)
    y = y * (5.12 / 100.0)
    y = rotatefunc(y, Mr1)
    y = oszfunc(y)
    y = asyfunc(y, beta)
    y = rotatefunc(y, Mr2)
    cond = np.power(alpha, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr1)  # third rotation with first matrix (matches C code)
    f = np.sum(y**2 - 10.0 * np.cos(2.0 * PI * y) + 10.0)
    return f

def step_rastrigin_func(x, Os, Mr1, Mr2):
    alpha = 10.0
    beta = 0.2
    y = shiftfunc(x, Os)
    y = y * (5.12 / 100.0)
    y = rotatefunc(y, Mr1)
    # step transformation
    y = np.where(np.abs(y) > 0.5, np.floor(2*y + 0.5)/2.0, y)
    y = oszfunc(y)
    y = asyfunc(y, beta)
    y = rotatefunc(y, Mr2)
    cond = np.power(alpha, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr1)   # third rotation with Mr1
    f = np.sum(y**2 - 10.0 * np.cos(2.0 * PI * y) + 10.0)
    return f

def schwefel_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = y * (1000.0 / 100.0)
    y = rotatefunc(y, Mr1)
    cond = np.power(10.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = y + 420.9687462275036
    f = 0.0
    for i in range(len(y)):
        if y[i] > 500:
            f -= (500.0 - math.fmod(y[i], 500)) * math.sin(math.sqrt(abs(500.0 - math.fmod(y[i], 500))))
            tmp = (y[i] - 500.0) / 100.0
            f += tmp * tmp / len(y)
        elif y[i] < -500:
            f -= (-500.0 + math.fmod(abs(y[i]), 500)) * math.sin(math.sqrt(abs(500.0 - math.fmod(abs(y[i]), 500))))
            tmp = (y[i] + 500.0) / 100.0
            f += tmp * tmp / len(y)
        else:
            f -= y[i] * math.sin(math.sqrt(abs(y[i])))
    return 418.9828872724338 * len(y) + f

def katsuura_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = y * (5.0 / 100.0)
    y = rotatefunc(y, Mr1)
    cond = np.power(100.0, 1.0 * np.arange(len(y)) / (len(y) - 1) / 2.0)
    y = y * cond
    y = rotatefunc(y, Mr2)
    n = len(y)
    tmp3 = n ** 1.2
    prod = 1.0
    for i in range(n):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * y[i]
            temp += abs(tmp2 - math.floor(tmp2 + 0.5)) / tmp1
        prod *= (1.0 + (i+1) * temp) ** (10.0 / tmp3)
    return (10.0 / (n * n)) * prod - (10.0 / (n * n))

def bi_rastrigin_func(x, Os, Mr1, Mr2):
    n = len(x)
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(n + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)

    y = shiftfunc(x, Os)
    y = y * (10.0 / 100.0)

    # Element-wise sign flip based on Os (matches C code exactly)
    tmpx = 2.0 * y.copy()
    for i in range(n):
        if Os[i] < 0.0:
            tmpx[i] *= -1.0

    z = tmpx.copy()
    tmpx = tmpx + mu0

    y_rot = rotatefunc(z, Mr1)
    cond = np.power(100.0, 1.0 * np.arange(n) / max(n - 1, 1) / 2.0)
    y_rot = y_rot * cond
    z_rot = rotatefunc(y_rot, Mr2)

    tmp1 = np.sum((tmpx - mu0) ** 2)
    tmp2 = s * np.sum((tmpx - mu1) ** 2) + d * n
    tmp = np.sum(np.cos(2.0 * PI * z_rot))

    f = min(tmp1, tmp2) + 10.0 * (n - tmp)
    return f

def grie_rosen_func(x, Os, Mr):
    y = shiftfunc(x, Os)
    y = y * (5.0 / 100.0)
    y = rotatefunc(y, Mr)
    y = y + 1.0
    n = len(y)
    f = 0.0
    for i in range(n-1):
        tmp1 = y[i]**2 - y[i+1]
        tmp2 = y[i] - 1.0
        temp = 100.0 * tmp1**2 + tmp2**2
        f += (temp**2)/4000.0 - math.cos(temp) + 1.0
    # last element with first
    tmp1 = y[n-1]**2 - y[0]
    tmp2 = y[n-1] - 1.0
    temp = 100.0 * tmp1**2 + tmp2**2
    f += (temp**2)/4000.0 - math.cos(temp) + 1.0
    return f

def escaffer6_func(x, Os, Mr1, Mr2):
    y = shiftfunc(x, Os)
    y = rotatefunc(y, Mr1)
    y = asyfunc(y, 0.5)
    y = rotatefunc(y, Mr2)
    n = len(y)
    f = 0.0
    for i in range(n-1):
        temp1 = math.sin(math.sqrt(y[i]**2 + y[i+1]**2))
        temp1 = temp1**2
        temp2 = 1.0 + 0.001 * (y[i]**2 + y[i+1]**2)
        f += 0.5 + (temp1 - 0.5) / (temp2**2)
    # wrap around
    temp1 = math.sin(math.sqrt(y[n-1]**2 + y[0]**2))
    temp1 = temp1**2
    temp2 = 1.0 + 0.001 * (y[n-1]**2 + y[0]**2)
    f += 0.5 + (temp1 - 0.5) / (temp2**2)
    return f

# ----------------------------------------------------------------------
# Composition functions (cf01 .. cf08)
# ----------------------------------------------------------------------
def cf01(x, nx, OShift, M, rotate):
    cf_num = 5
    delta = [10, 20, 30, 40, 50]
    bias  = [0, 100, 200, 300, 400]
    fit = np.zeros(cf_num)

    fit[0] = rosenbrock_func(x, OShift[0], M[0]) * 10000.0 / 1e4
    fit[1] = dif_powers_func(x, OShift[1], M[1]) * 10000.0 / 1e10
    fit[2] = bent_cigar_func(x, OShift[2], M[2], M[3]) * 10000.0 / 1e30  # needs two rotations
    fit[3] = discus_func(x, OShift[3], M[3]) * 10000.0 / 1e10
    fit[4] = sphere_func(x, OShift[4], M[4]) * 10000.0 / 1e5

    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf02(x, nx, OShift, M, rotate):
    cf_num = 3
    delta = [20, 20, 20]
    bias  = [0, 100, 200]
    fit = np.zeros(cf_num)
    for i in range(cf_num):
        fit[i] = schwefel_func(x, OShift[i], M[i*2], M[i*2+1])  # note: needs two matrices
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf03(x, nx, OShift, M, rotate):
    # identical to cf02 in CEC13, used for both func 22 (rotate=0) and 23 (rotate=1)
    cf_num = 3
    delta = [20, 20, 20]
    bias  = [0, 100, 200]
    fit = np.zeros(cf_num)
    for i in range(cf_num):
        fit[i] = schwefel_func(x, OShift[i], M[i*2], M[i*2+1])
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf04(x, nx, OShift, M, rotate):
    cf_num = 3
    delta = [20, 20, 20]
    bias  = [0, 100, 200]
    fit = np.zeros(cf_num)
    fit[0] = schwefel_func(x, OShift[0], M[0], M[1]) * 1000.0 / 4e3
    fit[1] = rastrigin_func(x, OShift[1], M[1], M[2]) * 1000.0 / 1e3
    fit[2] = weierstrass_func(x, OShift[2], M[2], M[3]) * 1000.0 / 400
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf05(x, nx, OShift, M, rotate):
    cf_num = 3
    delta = [10, 30, 50]
    bias  = [0, 100, 200]
    fit = np.zeros(cf_num)
    fit[0] = schwefel_func(x, OShift[0], M[0], M[1]) * 1000.0 / 4e3
    fit[1] = rastrigin_func(x, OShift[1], M[1], M[2]) * 1000.0 / 1e3
    fit[2] = weierstrass_func(x, OShift[2], M[2], M[3]) * 1000.0 / 400
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf06(x, nx, OShift, M, rotate):
    cf_num = 5
    delta = [10, 10, 10, 10, 10]
    bias  = [0, 100, 200, 300, 400]
    fit = np.zeros(cf_num)
    fit[0] = schwefel_func(x, OShift[0], M[0], M[1]) * 1000.0 / 4e3
    fit[1] = rastrigin_func(x, OShift[1], M[1], M[2]) * 1000.0 / 1e3
    fit[2] = ellips_func(x, OShift[2], M[2]) * 1000.0 / 1e10
    fit[3] = weierstrass_func(x, OShift[3], M[3], M[4]) * 1000.0 / 400
    fit[4] = griewank_func(x, OShift[4], M[4], M[5]) * 1000.0 / 100
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf07(x, nx, OShift, M, rotate):
    cf_num = 5
    delta = [10, 10, 10, 20, 20]
    bias  = [0, 100, 200, 300, 400]
    fit = np.zeros(cf_num)
    fit[0] = griewank_func(x, OShift[0], M[0], M[1]) * 10000.0 / 100
    fit[1] = rastrigin_func(x, OShift[1], M[1], M[2]) * 10000.0 / 1e3
    fit[2] = schwefel_func(x, OShift[2], M[2], M[3]) * 10000.0 / 4e3
    fit[3] = weierstrass_func(x, OShift[3], M[3], M[4]) * 10000.0 / 400
    fit[4] = sphere_func(x, OShift[4], M[4]) * 10000.0 / 1e5
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

def cf08(x, nx, OShift, M, rotate):
    cf_num = 5
    delta = [10, 20, 30, 40, 50]
    bias  = [0, 100, 200, 300, 400]
    fit = np.zeros(cf_num)
    fit[0] = grie_rosen_func(x, OShift[0], M[0]) * 10000.0 / 4e3
    fit[1] = schaffer_F7_func(x, OShift[1], M[1], M[2]) * 10000.0 / 4e6
    fit[2] = schwefel_func(x, OShift[2], M[2], M[3]) * 10000.0 / 4e3
    fit[3] = escaffer6_func(x, OShift[3], M[3], M[4]) * 10000.0 / 2e7
    fit[4] = sphere_func(x, OShift[4], M[4]) * 10000.0 / 1e5
    return cf_cal(x, nx, OShift[:cf_num], delta, bias, fit, cf_num)

# ----------------------------------------------------------------------
# Final dispatcher (internal)
# ----------------------------------------------------------------------
def _evaluate_point(x, func_num, nx, M, OShift):
    if func_num == 1:
        return sphere_func(x, OShift[0], M[0]) - 1400.0
    elif func_num == 2:
        return ellips_func(x, OShift[1], M[1]) - 1300.0
    elif func_num == 3:
        return bent_cigar_func(x, OShift[2], M[2], M[3]) - 1200.0
    elif func_num == 4:
        return discus_func(x, OShift[3], M[3]) - 1100.0
    elif func_num == 5:
        return dif_powers_func(x, OShift[4], M[4]) - 1000.0
    elif func_num == 6:
        return rosenbrock_func(x, OShift[5], M[5]) - 900.0
    elif func_num == 7:
        return schaffer_F7_func(x, OShift[6], M[6], M[7]) - 800.0
    elif func_num == 8:
        return ackley_func(x, OShift[7], M[7], M[8]) - 700.0
    elif func_num == 9:
        return weierstrass_func(x, OShift[8], M[8], M[9]) - 600.0
    elif func_num == 10:
        return griewank_func(x, OShift[9], M[9], M[0]) - 500.0
    elif func_num == 11:
        return rastrigin_func(x, OShift[0], M[0], M[1]) - 400.0
    elif func_num == 12:
        return rastrigin_func(x, OShift[1], M[1], M[2]) - 300.0
    elif func_num == 13:
        return step_rastrigin_func(x, OShift[2], M[2], M[3]) - 200.0
    elif func_num == 14:
        return schwefel_func(x, OShift[3], M[3], M[4]) - 100.0
    elif func_num == 15:
        return schwefel_func(x, OShift[4], M[4], M[5]) + 100.0
    elif func_num == 16:
        return katsuura_func(x, OShift[5], M[5], M[6]) + 200.0
    elif func_num == 17:
        return bi_rastrigin_func(x, OShift[6], M[6], M[7]) + 300.0
    elif func_num == 18:
        return bi_rastrigin_func(x, OShift[7], M[7], M[8]) + 400.0
    elif func_num == 19:
        return grie_rosen_func(x, OShift[8], M[8]) + 500.0
    elif func_num == 20:
        return escaffer6_func(x, OShift[9], M[9], M[0]) + 600.0
    elif func_num == 21:
        return cf01(x, nx, OShift, M, rotate=True) + 700.0
    elif func_num == 22:
        return cf02(x, nx, OShift, M, rotate=False) + 800.0
    elif func_num == 23:
        return cf03(x, nx, OShift, M, rotate=True) + 900.0
    elif func_num == 24:
        return cf04(x, nx, OShift, M, rotate=True) + 1000.0
    elif func_num == 25:
        return cf05(x, nx, OShift, M, rotate=True) + 1100.0
    elif func_num == 26:
        return cf06(x, nx, OShift, M, rotate=True) + 1200.0
    elif func_num == 27:
        return cf07(x, nx, OShift, M, rotate=True) + 1300.0
    elif func_num == 28:
        return cf08(x, nx, OShift, M, rotate=True) + 1400.0
    else:
        raise ValueError('Function number must be 1..28')

# ============================================================================
# CEC2013-compatible wrapper functions f1-f28
# Each takes a 1D numpy array x and returns a float
# ============================================================================

def _make_fi(func_num, bias):
    """Factory to create f1..f28 wrappers."""
    def fi(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        nx = len(x)
        M, OShift = _CEC13Data.load(nx)
        return float(_evaluate_point(x, func_num, nx, M, OShift))
    fi.__doc__ = f"""F{func_num}: CEC2013 Function {func_num} (optimal = {bias})"""
    fi.__name__ = f"f{func_num}"
    return fi


# Optimal values (biases) for CEC2013 F1-F28
_BIASES = {
    1: -1400, 2: -1300, 3: -1200, 4: -1100, 5: -1000,
    6: -900, 7: -800, 8: -700, 9: -600, 10: -500,
    11: -400, 12: -300, 13: -200, 14: -100, 15: 100,
    16: 200, 17: 300, 18: 400, 19: 500, 20: 600,
    21: 700, 22: 800, 23: 900, 24: 1000, 25: 1100,
    26: 1200, 27: 1300, 28: 1400,
}

# Generate f1 through f28
f1 = _make_fi(1, -1400)
f2 = _make_fi(2, -1300)
f3 = _make_fi(3, -1200)
f4 = _make_fi(4, -1100)
f5 = _make_fi(5, -1000)
f6 = _make_fi(6, -900)
f7 = _make_fi(7, -800)
f8 = _make_fi(8, -700)
f9 = _make_fi(9, -600)
f10 = _make_fi(10, -500)
f11 = _make_fi(11, -400)
f12 = _make_fi(12, -300)
f13 = _make_fi(13, -200)
f14 = _make_fi(14, -100)
f15 = _make_fi(15, 100)
f16 = _make_fi(16, 200)
f17 = _make_fi(17, 300)
f18 = _make_fi(18, 400)
f19 = _make_fi(19, 500)
f20 = _make_fi(20, 600)
f21 = _make_fi(21, 700)
f22 = _make_fi(22, 800)
f23 = _make_fi(23, 900)
f24 = _make_fi(24, 1000)
f25 = _make_fi(25, 1100)
f26 = _make_fi(26, 1200)
f27 = _make_fi(27, 1300)
f28 = _make_fi(28, 1400)


# ============================================================================
# Complete Function Dictionary (CEC2013-compatible interface)
# ============================================================================

ALL_FUNCTIONS = {
    i: {"objective": globals()[f"f{i}"], "lb": -100, "ub": 100, "g": [], "h": []}
    for i in range(1, 29)
}


def get_function(func_num: int) -> dict:
    """Get function info by number (1-28)"""
    return ALL_FUNCTIONS.get(func_num)


def evaluate(func_num: int, x: np.ndarray) -> float:
    """Evaluate a CEC2013 benchmark function"""
    func_info = get_function(func_num)
    if func_info is None:
        raise ValueError(f"Invalid function number: {func_num}. Must be 1-28")
    return func_info["objective"](x)