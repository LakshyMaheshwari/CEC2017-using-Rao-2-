"""
All 30 CEC2017 Benchmark Functions
Official CEC2017 benchmark functions implementation.

"""

import numpy as np
from typing import Optional, List, Callable
from .data_loader import (
    generate_rotation_matrix, generate_shift_vector, generate_shuffle_vector,
    generate_rotation_matrices, generate_shift_vectors
)

# ============================================================================
# Basic Function Definitions (from cec2017.basic)
# ============================================================================


def bent_cigar(x: np.ndarray) -> np.ndarray:
    """Bent Cigar function (basic)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    sm = np.sum(x[:, 1:] * x[:, 1:], axis=1)
    sm = sm * 1e6
    return x[:, 0] * x[:, 0] + sm


def zakharov(x: np.ndarray) -> np.ndarray:
    """Zakharov function (basic)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    i = np.expand_dims(np.arange(x.shape[1]) + 1, 0)
    sm = np.sum(i * x, axis=1)
    sms = np.sum(x * x, axis=1)
    sm = 0.5 * sm
    sm = sm * sm
    return sms + sm + (sm * sm)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock function (basic) - scaled for CEC2017"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = 0.02048 * x + 1.0
    t1 = x[:, :-1] * x[:, :-1] - x[:, 1:]
    t1 = 100 * t1 * t1
    t2 = x[:, :-1] - 1
    t2 = t2 * t2
    return np.sum(t1 + t2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """Rastrigin function (basic) - scaled for CEC2017"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = 0.0512 * x
    cs = np.cos(2 * np.pi * x)
    xs = x * x - 10 * cs + 10
    return np.sum(xs, axis=1)


def expanded_schaffers_f6(x: np.ndarray) -> np.ndarray:
    """Expanded Schaffer's F6 function (basic)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    t = x[:, :-1] * x[:, :-1] + x[:, 1:] * x[:, 1:]
    t1 = np.sin(np.sqrt(t))
    t1 = t1 * t1 - 0.5
    t2 = 1 + 0.001 * t
    t2 = t2 * t2
    return np.sum(0.5 + t1 / t2, axis=1)


def lunacek_bi_rastrigin(
    x: np.ndarray,
    shift: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Lunacek bi-Rastrigin function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if shift is None:
        shift = np.zeros((1, nx))
    else:
        shift = np.expand_dims(shift, 0)

    mu0 = 2.5
    s = 1 - 1 / (2 * ((nx + 20) ** 0.5) - 8.2)
    mu1 = -((mu0 * mu0 - 1) / s) ** 0.5

    y = 0.1 * (x - shift)
    tmpx = 2 * y
    tmpx[:, shift[0] < 0] *= -1
    z = tmpx.copy()
    tmpx = tmpx + mu0

    t1 = tmpx - mu0
    t1 = t1 * t1
    t1 = np.sum(t1, axis=1)
    t2 = tmpx - mu1
    t2 = s * t2 * t2
    t2 = np.sum(t2, axis=1) + nx

    if rotation is None:
        y = z
    else:
        y = np.matmul(
            np.expand_dims(rotation, 0),
            np.expand_dims(z, -1),
        )[:, :, 0]

    y = np.cos(2.0 * np.pi * y)
    t = np.sum(y, axis=1)

    r = t1
    r[t1 >= t2] = t2[t1 >= t2]
    return r + 10.0 * (nx - t)


def non_cont_rastrigin(
    x: np.ndarray,
    shift: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Non-Continuous Rastrigin function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if shift is None:
        shift = np.zeros((1, nx))
    else:
        shift = np.expand_dims(shift, 0)

    shifted = x - shift
    x = x.copy()
    mask = np.abs(shifted) > 0.5
    x[mask] = (shift + np.floor(2 * shifted + 0.5) * 0.5)[mask]

    z = 0.0512 * shifted
    if rotation is not None:
        z = np.matmul(
            np.expand_dims(rotation, 0),
            np.expand_dims(z, -1),
        )[:, :, 0]

    sm = z * z - 10 * np.cos(2 * np.pi * z) + 10
    sm = np.sum(sm, axis=1)
    return sm


def levy(x: np.ndarray) -> np.ndarray:
    """Levy function (basic)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    w = 1.0 + 0.25 * (x - 1.0)

    term1 = (np.sin(np.pi * w[:, 0])) ** 2
    term3 = ((w[:, -1] - 1) ** 2) * (1 + ((np.sin(2 * np.pi * w[:, -1])) ** 2))

    wi = w[:, :-1]
    newv = ((wi - 1) ** 2) * (1 + 10 * ((np.sin(np.pi * wi + 1)) ** 2))
    sm = np.sum(newv, axis=1)

    return term1 + sm + term3


def modified_schwefel(x: np.ndarray) -> np.ndarray:
    """Modified Schwefel function (basic)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]
    x = 10.0 * x

    z = x + 420.9687462275036
    mask1 = z < -500
    mask2 = z > 500
    sm = z * np.sin(np.sqrt(np.abs(z)))

    zm = np.mod(np.abs(z), 500)
    zm[mask1] = (zm[mask1] - 500)
    zm[mask2] = (500 - zm[mask2])
    t = z + 500
    t[mask2] = z[mask2] - 500
    t = t * t

    mask1_or_2 = np.logical_or(mask1, mask2)
    sm[mask1_or_2] = (zm * np.sin(np.sqrt(np.abs(zm))) - t / (10_000 * nx))[mask1_or_2]
    return 418.9829 * nx - np.sum(sm, axis=1)


def high_conditioned_elliptic(x: np.ndarray) -> np.ndarray:
    """High Conditioned Elliptic function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Handle edge case where dimension is too small
    if x.shape[1] <= 1:
        return np.sum(x * x, axis=1)
    factor = 6 / (x.shape[1] - 1)
    i = np.expand_dims(np.arange(x.shape[1]), 0)
    sm = x * x * 10 ** (i * factor)
    return np.sum(sm, axis=1)


def discus(x: np.ndarray) -> np.ndarray:
    """Discus function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    sm0 = 1e+6 * x[:, 0] * x[:, 0]
    sm = np.sum(x[:, 1:] * x[:, 1:], axis=1)
    return sm0 + sm


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    smsq = np.sum(x * x, axis=1)
    smcs = np.sum(np.cos((2 * np.pi) * x), axis=1)
    inx = 1 / x.shape[1]
    return -20 * np.exp(-0.2 * np.sqrt(inx * smsq)) - np.exp(inx * smcs) + 20 + np.e


def weierstrass(x: np.ndarray) -> np.ndarray:
    """Weierstrass function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = 0.005 * x
    k = np.arange(start=0, stop=21, step=1)
    k = np.expand_dims(np.expand_dims(k, 0), 0)
    ak = 0.5 ** k
    bk = np.pi * (3 ** k)

    kcs = ak * np.cos(2 * (np.expand_dims(x, -1) + 0.5) * bk)
    ksm = np.sum(kcs, axis=2)
    sm = np.sum(ksm, axis=1)

    kcs = ak * np.cos(bk)
    ksm = np.sum(kcs)
    return sm - x.shape[1] * ksm


def griewank(x: np.ndarray) -> np.ndarray:
    """Griewank function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]
    x = 6.0 * x
    factor = 1 / 4000
    d = np.expand_dims(np.arange(start=1, stop=nx + 1), 0)
    np.cos(x / d)
    sm = np.sum(factor * x * x, axis=1)
    pd = np.prod(np.cos(x / d), axis=1)
    return sm - pd + 1


def katsuura(x: np.ndarray) -> np.ndarray:
    """Katsuura function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = 0.05 * x
    nx = x.shape[1]
    # Handle edge case where dimension is too small
    if nx <= 1:
        return np.sum(x * x, axis=1)
    pw = 10 / (nx ** 1.2)
    prd = 1.0
    tj = 2 ** np.arange(start=1, stop=33, step=1, dtype=np.int64)
    tj = np.expand_dims(np.expand_dims(tj, 0), 0)
    tjx = tj * np.expand_dims(x, -1)
    t = np.abs(tjx - np.round(tjx)) / tj
    tsm = np.sum(t, axis=2)

    i = np.arange(nx) + 1
    prd = np.prod((1 + i * tsm) ** pw, axis=1)
    df = 10 / (nx * nx)
    return df * prd - df


def happy_cat(x: np.ndarray) -> np.ndarray:
    """HappyCat function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = (0.05 * x) - 1
    nx = x.shape[1]
    sm = np.sum(x, axis=1)
    smsq = np.sum(x * x, axis=1)
    return (np.abs(smsq - nx)) ** 0.25 + (0.5 * smsq + sm) / nx + 0.5


def h_g_bat(x: np.ndarray) -> np.ndarray:
    """HGBat function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = (0.05 * x) - 1
    nx = x.shape[1]
    sm = np.sum(x, axis=1)
    smsq = np.sum(x * x, axis=1)
    return (np.abs(smsq * smsq - sm * sm)) ** 0.5 + (0.5 * smsq + sm) / nx + 0.5


def expanded_griewanks_plus_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Expanded Griewank's plus Rosenbrock function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Handle edge case where dimension is too small
    if x.shape[1] <= 1:
        return np.sum(x * x, axis=1)
    x = (0.05 * x) + 1

    tmp1 = x[:, :-1] * x[:, :-1] - x[:, 1:]
    tmp2 = x[:, :-1] - 1.0
    temp = 100 * tmp1 * tmp1 + tmp2 * tmp2
    sm = (temp * temp) / 4000 - np.cos(temp) + 1

    tmp1 = x[:, -1:] * x[:, -1:] - x[:, 0:1]
    tmp2 = x[:, -1:] - 1
    temp = 100 * tmp1 * tmp1 + tmp2 * tmp2
    sm = sm + (temp * temp) / 4000 - np.cos(temp) + 1

    return np.sum(sm, axis=1)


def schaffers_f7(x: np.ndarray) -> np.ndarray:
    """Schaffer's F7 function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]
    si = np.sqrt(x[:, :-1] * x[:, :-1] + x[:, 1:] * x[:, 1:])
    tmp = np.sin(50 * (np.power(si, 0.2)))
    sm = np.sqrt(si) * (tmp * tmp + 1)
    sm = np.sum(sm, axis=1)
    sm = (sm * sm) / (nx * nx - 2 * nx + 1)
    return sm


# ============================================================================
# Simple Functions F1-F10 (Shifted and Rotated)
# ============================================================================

def shift_rotate(x: np.ndarray, shift: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply shift and rotation to vector x"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    shifted = np.expand_dims(x - np.expand_dims(shift, 0), -1)
    x_transformed = np.matmul(np.expand_dims(rotation, 0), shifted)
    return x_transformed[:, :, 0]


def f1(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F1: Shifted and Rotated Bent Cigar Function"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(1, nx)
    if shift is None:
        shift = generate_shift_vector(1, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    return float(bent_cigar(x_transformed)[0] + 100.0)


def f2(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F2: (Deprecated from CEC2017)"""
    print("WARNING: f2 has been deprecated from the CEC 2017 benchmark suite")
    return 200.0


def f3(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F3: Shifted and Rotated Zakharov Function"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(3, nx)
    if shift is None:
        shift = generate_shift_vector(3, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    return float(zakharov(x_transformed)[0] + 300.0)


def f4(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F4: Shifted and Rotated Rosenbrock's Function"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(4, nx)
    if shift is None:
        shift = generate_shift_vector(4, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    return float(rosenbrock(x_transformed)[0] + 400.0)


def f5(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F5: Shifted and Rotated Rastrigin's Function"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(5, nx)
    if shift is None:
        shift = generate_shift_vector(5, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    return float(rastrigin(x_transformed)[0] + 500.0)


def f6(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None) -> float:
    """F6: Shifted and Rotated Schaffer's F7 Function
    Official: f20(M * (0.5*(x-o6)/100)) + 600
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(6, nx)
    if shift is None:
        shift = generate_shift_vector(6, nx)

    scaled = 0.5 * (x - np.expand_dims(shift, 0)) / 100.0
    x_transformed = np.matmul(np.expand_dims(rotation, 0), np.expand_dims(scaled, -1))[:, :, 0]
    return float(schaffers_f7(x_transformed)[0] + 600.0)


def f7(x: np.ndarray, rotation: Optional[np.ndarray] = None,
       shift: Optional[np.ndarray] = None) -> float:
    """F7: Shifted and Rotated Lunacek Bi-Rastrigin's Function
    Official: F7(x) = f7( M7 * (600*(x-o7)/100) ) + 700
    Rotation must be applied to the scaled-shifted vector BEFORE
    passing to lunacek. lunacek internally does y = 0.1*(x-shift),
    so we pass the rotated-scaled result as x, with shift=0.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(7, nx)
    if shift is None:
        shift = generate_shift_vector(7, nx)

    # Step 1: shift and scale
    z = (x - np.expand_dims(shift, 0)) * (600.0 / 100.0)

    # Step 2: apply rotation
    z_rot = np.matmul(np.expand_dims(rotation, 0),
                      np.expand_dims(z, -1))[:, :, 0]

    # Step 3: pass to lunacek with zero shift — transformation already done
    zero_shift = np.zeros(nx)
    return float(lunacek_bi_rastrigin(z_rot, zero_shift, None)[0] + 700.0)


def f8(x: np.ndarray, rotation: Optional[np.ndarray] = None,
       shift: Optional[np.ndarray] = None) -> float:
    """F8: Shifted and Rotated Non-Continuous Rastrigin's Function
    Official: F8(x) = f8( 5.12*(x-o8)/100 ) + 800
    non_cont_rastrigin internally applies 0.0512 scaling — do NOT pre-scale.
    The 5.12/100 outer scale IS the search range normalisation.
    non_cont_rastrigin expects raw shifted x and handles scaling internally.
    So we just pass shift and rotation — no manual pre-scaling.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(8, nx)
    if shift is None:
        shift = generate_shift_vector(8, nx)

    # Pass raw x — non_cont_rastrigin handles shift, scaling, and rotation
    return float(non_cont_rastrigin(x, shift, rotation)[0] + 800.0)


def f9(x: np.ndarray, rotation: Optional[np.ndarray] = None,
       shift: Optional[np.ndarray] = None) -> float:
    """F9: Shifted and Rotated Levy Function
    Official: F9(x) = f9( M9 * (5.12*(x-o9)/100) ) + 900
    levy has NO internal scaling — this is correct as-is.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(9, nx)
    if shift is None:
        shift = generate_shift_vector(9, nx)

    # Scale → rotate → levy
    scaled = 5.12 * (x - np.expand_dims(shift, 0)) / 100.0
    x_transformed = np.matmul(np.expand_dims(rotation, 0),
                              np.expand_dims(scaled, -1))[:, :, 0]
    return float(levy(x_transformed)[0] + 900.0)


def f10(x: np.ndarray, rotation: Optional[np.ndarray] = None,
        shift: Optional[np.ndarray] = None) -> float:
    """F10: Shifted and Rotated Schwefel's Function
    Official: F10(x) = f10( 1000*(x-o10)/100 ) + 1000
    modified_schwefel internally multiplies by 10 and adds 420.96.
    So 1000/100 = 10, then *10 inside = 100x — that is DOUBLE scaling.
    Correct outer scale is 100/100 = 1.0 so modified_schwefel's
    internal *10 brings it to the right 10x total.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nx = x.shape[1]
    if rotation is None:
        rotation = generate_rotation_matrix(10, nx)
    if shift is None:
        shift = generate_shift_vector(10, nx)

    # Correct scale: 100/100 = 1.0 so modified_schwefel's internal *10
    # produces the correct 1000/100 = 10x total scaling
    scaled = (x - np.expand_dims(shift, 0)) * (100.0 / 100.0)
    return float(modified_schwefel(scaled)[0] + 1000.0)

# ============================================================================
# Hybrid Functions F11-F20
# ============================================================================


def shuffle_and_partition(x: np.ndarray, shuffle: np.ndarray, partitions: List[float]) -> List[np.ndarray]:
    """Shuffle and partition vector"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    xs = np.zeros_like(x)
    for i in range(nx):
        xs[:, i] = x[:, shuffle[i]]

    parts = []
    start, end = 0, 0
    for p in partitions[:-1]:
        end = start + int(np.ceil(p * nx))
        parts.append(xs[:, start:end])
        start = end
    parts.append(xs[:, end:])
    return parts


def f11(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F11: Hybrid Function 1 (N=3)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(11, nx)
    if shift is None:
        shift = generate_shift_vector(11, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(11, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.4, 0.4])

    y = zakharov(x_parts[0])[0]
    y += rosenbrock(x_parts[1])[0]
    y += rastrigin(x_parts[2])[0]
    return float(y + 1100.0)


def f12(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F12: Hybrid Function 2 (N=3)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if nx < 3:
        if rotation is None:
            rotation = generate_rotation_matrix(12, nx)
        if shift is None:
            shift = generate_shift_vector(12, nx)
        x_transformed = shift_rotate(x, shift, rotation)
        y = high_conditioned_elliptic(x_transformed)[0]
        y += modified_schwefel(x_transformed)[0]
        y += bent_cigar(x_transformed)[0]
        return float(y + 1200.0)

    if rotation is None:
        rotation = generate_rotation_matrix(12, nx)
    if shift is None:
        shift = generate_shift_vector(12, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(12, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = high_conditioned_elliptic(x_parts[0])[0]
    y += modified_schwefel(x_parts[1])[0]
    y += bent_cigar(x_parts[2])[0]
    return float(y + 1200.0)


def f13(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F13: Hybrid Function 3 (N=3)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(13, nx)
    if shift is None:
        shift = generate_shift_vector(13, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(13, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = bent_cigar(x_parts[0])[0]
    y += rosenbrock(x_parts[1])[0]
    y += lunacek_bi_rastrigin(x_parts[2])[0]
    return float(y + 1300.0)


def f14(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F14: Hybrid Function 4 (N=4)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if nx < 4:
        if rotation is None:
            rotation = generate_rotation_matrix(14, nx)
        if shift is None:
            shift = generate_shift_vector(14, nx)
        x_transformed = shift_rotate(x, shift, rotation)
        y = high_conditioned_elliptic(x_transformed)[0]
        y += ackley(x_transformed)[0]
        y += schaffers_f7(x_transformed)[0]
        y += rastrigin(x_transformed)[0]
        return float(y + 1400.0)

    if rotation is None:
        rotation = generate_rotation_matrix(14, nx)
    if shift is None:
        shift = generate_shift_vector(14, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(14, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.4])

    y = high_conditioned_elliptic(x_parts[0])[0]
    y += ackley(x_parts[1])[0]
    y += schaffers_f7(x_parts[2])[0]
    y += rastrigin(x_parts[3])[0]
    return float(y + 1400.0)


def f15(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F15: Hybrid Function 5 (N=4)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(15, nx)
    if shift is None:
        shift = generate_shift_vector(15, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(15, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = bent_cigar(x_parts[0])[0]
    y += h_g_bat(x_parts[1])[0]
    y += rastrigin(x_parts[2])[0]
    y += rosenbrock(x_parts[3])[0]
    return float(y + 1500.0)


def f16(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F16: Hybrid Function 6 (N=4)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(16, nx)
    if shift is None:
        shift = generate_shift_vector(16, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(16, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = expanded_schaffers_f6(x_parts[0])[0]
    y += h_g_bat(x_parts[1])[0]
    y += rosenbrock(x_parts[2])[0]
    y += modified_schwefel(x_parts[3])[0]
    return float(y + 1600.0)


def f17(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F17: Hybrid Function 7 (N=5)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(17, nx)
    if shift is None:
        shift = generate_shift_vector(17, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(17, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.1, 0.2, 0.2, 0.2, 0.3])

    y = katsuura(x_parts[0])[0]
    y += ackley(x_parts[1])[0]
    y += expanded_griewanks_plus_rosenbrock(x_parts[2])[0]
    y += modified_schwefel(x_parts[3])[0]
    y += rastrigin(x_parts[4])[0]
    return float(y + 1700.0)


def f18(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F18: Hybrid Function 8 (N=5)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if nx < 5:
        if rotation is None:
            rotation = generate_rotation_matrix(18, nx)
        if shift is None:
            shift = generate_shift_vector(18, nx)
        x_transformed = shift_rotate(x, shift, rotation)
        y = high_conditioned_elliptic(x_transformed)[0]
        y += ackley(x_transformed)[0]
        y += rastrigin(x_transformed)[0]
        y += h_g_bat(x_transformed)[0]
        y += discus(x_transformed)[0]
        return float(y + 1800.0)

    if rotation is None:
        rotation = generate_rotation_matrix(18, nx)
    if shift is None:
        shift = generate_shift_vector(18, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(18, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = high_conditioned_elliptic(x_parts[0])[0]
    y += ackley(x_parts[1])[0]
    y += rastrigin(x_parts[2])[0]
    y += h_g_bat(x_parts[3])[0]
    y += discus(x_parts[4])[0]
    return float(y + 1800.0)


def f19(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F19: Hybrid Function 9 (N=5)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if rotation is None:
        rotation = generate_rotation_matrix(19, nx)
    if shift is None:
        shift = generate_shift_vector(19, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(19, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = bent_cigar(x_parts[0])[0]
    y += rastrigin(x_parts[1])[0]
    y += expanded_griewanks_plus_rosenbrock(x_parts[2])[0]
    y += weierstrass(x_parts[3])[0]
    y += expanded_schaffers_f6(x_parts[4])[0]
    return float(y + 1900.0)


def f20(x: np.ndarray, rotation: Optional[np.ndarray] = None, shift: Optional[np.ndarray] = None,
        shuffle: Optional[np.ndarray] = None) -> float:
    """F20: Hybrid Function 10 (N=6)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    if nx < 6:
        if rotation is None:
            rotation = generate_rotation_matrix(20, nx)
        if shift is None:
            shift = generate_shift_vector(20, nx)
        x_transformed = shift_rotate(x, shift, rotation)
        y = happy_cat(x_transformed)[0]
        y += katsuura(x_transformed)[0]
        y += ackley(x_transformed)[0]
        y += rastrigin(x_transformed)[0]
        y += modified_schwefel(x_transformed)[0]
        y += schaffers_f7(x_transformed)[0]
        return float(y + 2000.0)

    if rotation is None:
        rotation = generate_rotation_matrix(20, nx)
    if shift is None:
        shift = generate_shift_vector(20, nx)
    if shuffle is None:
        shuffle = generate_shuffle_vector(20, nx)

    x_transformed = shift_rotate(x, shift, rotation)
    x_parts = shuffle_and_partition(x_transformed, shuffle, [0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

    y = happy_cat(x_parts[0])[0]
    y += katsuura(x_parts[1])[0]
    y += ackley(x_parts[2])[0]
    y += rastrigin(x_parts[3])[0]
    y += modified_schwefel(x_parts[4])[0]
    y += schaffers_f7(x_parts[5])[0]
    return float(y + 2000.0)


# ============================================================================
# Composition Functions F21-F30
# ============================================================================

def _calc_w(x: np.ndarray, sigma: float) -> np.ndarray:
    """Calculate weight for composition functions"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]
    w = np.sum(x * x, axis=1)
    nzmask = w != 0
    # Compute only on non-zero elements to avoid divide-by-zero
    w_nz = w[nzmask]
    w[nzmask] = (1.0 / w_nz) ** 0.5 * np.exp(-w_nz / (2.0 * nx * sigma * sigma))
    w[~nzmask] = float('inf')
    return w


def _composition(x: np.ndarray, rotations: List[np.ndarray], shifts: List[np.ndarray],
                 funcs: List[Callable], sigmas: List[float], lambdas: List[float],
                 biases: List[float]) -> np.ndarray:
    """Generic composition function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nv = x.shape[0]
    nx = x.shape[1]
    N = len(funcs)

    vals = np.zeros((nv, N))
    w = np.zeros((nv, N))

    for i in range(N):
        x_shifted = x - np.expand_dims(shifts[i][:nx], 0)
        x_t = shift_rotate(x, shifts[i][:nx], rotations[i])
        vals[:, i] = funcs[i](x_t)
        w[:, i] = _calc_w(x_shifted, sigmas[i])

    w_sm = np.sum(w, axis=1)
    # Handle rows where any weight is inf (x landed exactly on a shift vector)
    has_inf = np.any(np.isinf(w), axis=1)
    finite_mask = ~has_inf & (w_sm != 0.0)

    # Normal case: finite weights, normalize
    w[finite_mask, :] /= w_sm[finite_mask, None]

    # Inf case: set inf entries to 1, others to 0, then normalize
    for i in np.where(has_inf)[0]:
        inf_cols = np.isinf(w[i, :])
        n_inf = np.sum(inf_cols)
        w[i, :] = 0.0
        w[i, inf_cols] = 1.0 / n_inf

    # Zero case: equal weights
    zero_mask = ~has_inf & (w_sm == 0.0)
    w[zero_mask, :] = 1 / N

    return np.sum(w * (lambdas * vals + biases), axis=1)


def f21(x: np.ndarray) -> float:
    """F21: Composition Function 1 (N=3)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(21, nx, 3)
    shifts = generate_shift_vectors(21, nx, 3)
    funcs = [rosenbrock, high_conditioned_elliptic, rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2100.0)


def f22(x: np.ndarray) -> float:
    """F22: Composition Function 2 (N=3)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(22, nx, 3)
    shifts = generate_shift_vectors(22, nx, 3)
    funcs = [rastrigin, griewank, modified_schwefel]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2200.0)


def f23(x: np.ndarray) -> float:
    """F23: Composition Function 3 (N=4)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(23, nx, 4)
    shifts = generate_shift_vectors(23, nx, 4)
    funcs = [rosenbrock, ackley, modified_schwefel, rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 10.0, 1.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2300.0)


def f24(x: np.ndarray) -> float:
    """F24: Composition Function 4 (N=4)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(24, nx, 4)
    shifts = generate_shift_vectors(24, nx, 4)
    funcs = [ackley, high_conditioned_elliptic, griewank, rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 1.0e-6, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2400.0)


def f25(x: np.ndarray) -> float:
    """F25: Composition Function 5 (N=5)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(25, nx, 5)
    shifts = generate_shift_vectors(25, nx, 5)
    funcs = [rastrigin, happy_cat, ackley, discus, rosenbrock]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lambdas = np.array([10.0, 1.0, 10.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2500.0)


def f26(x: np.ndarray) -> float:
    """F26: Composition Function 6 (N=5)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(26, nx, 5)
    shifts = generate_shift_vectors(26, nx, 5)
    funcs = [expanded_schaffers_f6, modified_schwefel, griewank, rosenbrock, rastrigin]
    sigmas = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
    lambdas = np.array([5.0e-4, 1.0, 10.0, 1.0, 10.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2600.0)


def f27(x: np.ndarray) -> float:
    """F27: Composition Function 7 (N=6)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(27, nx, 6)
    shifts = generate_shift_vectors(27, nx, 6)
    funcs = [h_g_bat, rastrigin, modified_schwefel, bent_cigar, high_conditioned_elliptic, expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 2.5, 1.0e-26, 1.0e-6, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2700.0)


def f28(x: np.ndarray) -> float:
    """F28: Composition Function 8 (N=6)"""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(28, nx, 6)
    shifts = generate_shift_vectors(28, nx, 6)
    funcs = [ackley, griewank, discus, rosenbrock, happy_cat, expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 1.0e-6, 1.0, 1.0, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2800.0)

# ── Bare hybrid evaluators for use inside composition ─────────────────────
# These receive already-transformed input from _composition.
# They must NOT apply their own shift/rotation.


def _hybrid5_bare(x: np.ndarray) -> float:
    """F15 (Hybrid 5) — no internal shift/rotation."""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    # Small-dimension fallback: can't partition into 4 non-empty parts
    if nx < 4:
        y = bent_cigar(x)[0]
        y += h_g_bat(x)[0]
        y += rastrigin(x)[0]
        y += rosenbrock(x)[0]
        return float(y)

    shuffle = np.arange(nx)
    x_parts = shuffle_and_partition(x, shuffle, [0.2, 0.2, 0.3, 0.3])
    y = bent_cigar(x_parts[0])[0]
    y += h_g_bat(x_parts[1])[0]
    y += rastrigin(x_parts[2])[0]
    y += rosenbrock(x_parts[3])[0]
    return float(y)


def _hybrid6_bare(x: np.ndarray) -> float:
    """F16 (Hybrid 6) — no internal shift/rotation."""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    # Small-dimension fallback: can't partition into 4 non-empty parts
    if nx < 4:
        y = expanded_schaffers_f6(x)[0]
        y += h_g_bat(x)[0]
        y += rosenbrock(x)[0]
        y += modified_schwefel(x)[0]
        return float(y)

    shuffle = np.arange(nx)
    x_parts = shuffle_and_partition(x, shuffle, [0.2, 0.2, 0.3, 0.3])
    y = expanded_schaffers_f6(x_parts[0])[0]
    y += h_g_bat(x_parts[1])[0]
    y += rosenbrock(x_parts[2])[0]
    y += modified_schwefel(x_parts[3])[0]
    return float(y)


def _hybrid7_bare(x: np.ndarray) -> float:
    """F17 (Hybrid 7) — no internal shift/rotation."""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    # Small-dimension fallback: can't partition into 5 non-empty parts
    if nx < 5:
        y = katsuura(x)[0]
        y += ackley(x)[0]
        y += expanded_griewanks_plus_rosenbrock(x)[0]
        y += modified_schwefel(x)[0]
        y += rastrigin(x)[0]
        return float(y)

    shuffle = np.arange(nx)
    x_parts = shuffle_and_partition(x, shuffle, [0.1, 0.2, 0.2, 0.2, 0.3])
    y = katsuura(x_parts[0])[0]
    y += ackley(x_parts[1])[0]
    y += expanded_griewanks_plus_rosenbrock(x_parts[2])[0]
    y += modified_schwefel(x_parts[3])[0]
    y += rastrigin(x_parts[4])[0]
    return float(y)


def _hybrid8_bare(x: np.ndarray) -> float:
    """F18 (Hybrid 8) — no internal shift/rotation."""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    # Small-dimension fallback: can't partition into 5 non-empty parts
    if nx < 5:
        y = high_conditioned_elliptic(x)[0]
        y += ackley(x)[0]
        y += rastrigin(x)[0]
        y += h_g_bat(x)[0]
        y += discus(x)[0]
        return float(y)

    shuffle = np.arange(nx)
    x_parts = shuffle_and_partition(x, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])
    y = high_conditioned_elliptic(x_parts[0])[0]
    y += ackley(x_parts[1])[0]
    y += rastrigin(x_parts[2])[0]
    y += h_g_bat(x_parts[3])[0]
    y += discus(x_parts[4])[0]
    return float(y)


def _hybrid9_bare(x: np.ndarray) -> float:
    """F19 (Hybrid 9) — no internal shift/rotation."""
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    # Small-dimension fallback: can't partition into 5 non-empty parts
    if nx < 5:
        y = bent_cigar(x)[0]
        y += rastrigin(x)[0]
        y += expanded_griewanks_plus_rosenbrock(x)[0]
        y += weierstrass(x)[0]
        y += expanded_schaffers_f6(x)[0]
        return float(y)

    shuffle = np.arange(nx)
    x_parts = shuffle_and_partition(x, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])
    y = bent_cigar(x_parts[0])[0]
    y += rastrigin(x_parts[1])[0]
    y += expanded_griewanks_plus_rosenbrock(x_parts[2])[0]
    y += weierstrass(x_parts[3])[0]
    y += expanded_schaffers_f6(x_parts[4])[0]
    return float(y)


def f29(x: np.ndarray) -> float:
    """F29: Composition Function 9 (N=3)
    Components: Hybrid 5, Hybrid 8, Hybrid 9 — bare (no internal shift)
    sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(29, nx, 3)
    shifts = generate_shift_vectors(29, nx, 3)
    funcs = [_hybrid5_bare, _hybrid8_bare, _hybrid9_bare]
    sigmas = np.array([10.0, 30.0, 50.0])
    lambdas = np.array([1.0,  1.0,  1.0])
    biases = np.array([0.0,  100.0, 200.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 2900.0)


def f30(x: np.ndarray) -> float:
    """F30: Composition Function 10 (N=3)
    Components: Hybrid 5, Hybrid 6, Hybrid 7 — bare (no internal shift)
    sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    nx = x.shape[1]

    rotations = generate_rotation_matrices(30, nx, 3)
    shifts = generate_shift_vectors(30, nx, 3)
    funcs = [_hybrid5_bare, _hybrid6_bare, _hybrid7_bare]
    sigmas = np.array([10.0, 30.0, 50.0])
    lambdas = np.array([1.0,  1.0,  1.0])
    biases = np.array([0.0,  100.0, 200.0])

    return float(_composition(x, rotations, shifts, funcs, sigmas, lambdas, biases)[0] + 3000.0)

# ============================================================================
# Complete Function Dictionary
# ============================================================================


ALL_FUNCTIONS = {
    1: {"objective": f1, "lb": -100, "ub": 100, "g": [], "h": []},
    2: {"objective": f2, "lb": -100, "ub": 100, "g": [], "h": []},
    3: {"objective": f3, "lb": -100, "ub": 100, "g": [], "h": []},
    4: {"objective": f4, "lb": -100, "ub": 100, "g": [], "h": []},
    5: {"objective": f5, "lb": -100, "ub": 100, "g": [], "h": []},
    6: {"objective": f6, "lb": -100, "ub": 100, "g": [], "h": []},
    7: {"objective": f7, "lb": -100, "ub": 100, "g": [], "h": []},
    8: {"objective": f8, "lb": -100, "ub": 100, "g": [], "h": []},
    9: {"objective": f9, "lb": -100, "ub": 100, "g": [], "h": []},
    10: {"objective": f10, "lb": -100, "ub": 100, "g": [], "h": []},
    11: {"objective": f11, "lb": -100, "ub": 100, "g": [], "h": []},
    12: {"objective": f12, "lb": -100, "ub": 100, "g": [], "h": []},
    13: {"objective": f13, "lb": -100, "ub": 100, "g": [], "h": []},
    14: {"objective": f14, "lb": -100, "ub": 100, "g": [], "h": []},
    15: {"objective": f15, "lb": -100, "ub": 100, "g": [], "h": []},
    16: {"objective": f16, "lb": -100, "ub": 100, "g": [], "h": []},
    17: {"objective": f17, "lb": -100, "ub": 100, "g": [], "h": []},
    18: {"objective": f18, "lb": -100, "ub": 100, "g": [], "h": []},
    19: {"objective": f19, "lb": -100, "ub": 100, "g": [], "h": []},
    20: {"objective": f20, "lb": -100, "ub": 100, "g": [], "h": []},
    21: {"objective": f21, "lb": -100, "ub": 100, "g": [], "h": []},
    22: {"objective": f22, "lb": -100, "ub": 100, "g": [], "h": []},
    23: {"objective": f23, "lb": -100, "ub": 100, "g": [], "h": []},
    24: {"objective": f24, "lb": -100, "ub": 100, "g": [], "h": []},
    25: {"objective": f25, "lb": -100, "ub": 100, "g": [], "h": []},
    26: {"objective": f26, "lb": -100, "ub": 100, "g": [], "h": []},
    27: {"objective": f27, "lb": -100, "ub": 100, "g": [], "h": []},
    28: {"objective": f28, "lb": -100, "ub": 100, "g": [], "h": []},
    29: {"objective": f29, "lb": -100, "ub": 100, "g": [], "h": []},
    30: {"objective": f30, "lb": -100, "ub": 100, "g": [], "h": []},
}


def get_function(func_num: int) -> dict:
    """Get function info by number (1-30)"""
    return ALL_FUNCTIONS.get(func_num)


def evaluate(func_num: int, x: np.ndarray) -> float:
    """Evaluate a CEC2017 benchmark function"""
    func_info = get_function(func_num)
    if func_info is None:
        raise ValueError(f"Invalid function number: {func_num}. Must be 1-30")
    return func_info["objective"](x)
