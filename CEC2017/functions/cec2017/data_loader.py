import numpy as np
import os
from typing import List

_rotation_cache = {}
_shift_cache = {}
_shuffle_cache = {}
_rotation_matrices_cache = {}
_shift_vectors_cache = {}
_shuffle_vectors_cache = {}

# Default data directory: 'data' folder next to this file
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def generate_rotation_matrix(func_id: int, dimension: int) -> np.ndarray:
    """
    Reads the rotation matrix from M_{func_id}_D{dimension}.m or .txt
    """
    cache_key = (func_id, dimension)
    if cache_key in _rotation_cache:
        return _rotation_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"M_{func_id}_D{dimension}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"M_{func_id}_D{dimension}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[CEC2017 Error] Rotation matrix missing: '{filepath}'. Please ensure the official CEC2017 data files are in your DATA_DIR.")

    # Read the text file into a numpy array and reshape to DxD
    M = np.loadtxt(filepath)
    M = M.reshape((dimension, dimension))

    _rotation_cache[cache_key] = M
    return M


def generate_shift_vector(func_id: int, dimension: int) -> np.ndarray:
    """
    Reads the shift vector from shift_data_{func_id}.m or .txt
    Extracts only the first `dimension` values since the file contains up to 100 values.
    """
    cache_key = (func_id, dimension)
    if cache_key in _shift_cache:
        return _shift_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"shift_data_{func_id}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"shift_data_{func_id}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[CEC2017 Error] Shift vector missing: '{filepath}'. Please ensure the official CEC2017 data files are in your DATA_DIR.")

    # Read array from text
    shift = np.loadtxt(filepath)
    if shift.ndim == 2:
        shift = shift[0]  # Just in case there are multiple lines (like in composition files parsed incorrectly)

    # Official shift files contain up to 100 values. We truncate to the active dimension
    shift = shift[:dimension]

    _shift_cache[cache_key] = shift
    return shift


def generate_shuffle_vector(func_id: int, dimension: int) -> np.ndarray:
    """
    Reads the shuffle vector from shuffle_data_{func_id}_D{dimension}.m or .txt
    MATLAB indices are 1-based. Subtracts 1 to make them 0-based Python indices.
    """
    cache_key = (func_id, dimension)
    if cache_key in _shuffle_cache:
        return _shuffle_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"shuffle_data_{func_id}_D{dimension}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"shuffle_data_{func_id}_D{dimension}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[CEC2017 Error] Shuffle vector missing: '{filepath}'. Please ensure the official CEC2017 data files are in your DATA_DIR.")

    # Read an array of integers and convert from 1-based (MATLAB) to 0-based (Python)
    shuffle = np.loadtxt(filepath, dtype=int) - 1
    if shuffle.ndim == 2:
        shuffle = shuffle[0]

    _shuffle_cache[cache_key] = shuffle
    return shuffle


def generate_rotation_matrices(func_id: int, dimension: int, n_funcs: int) -> List[np.ndarray]:
    """
    Reads multiple rotation matrices from a stacked M_{func_id}_D{dimension} file 
    for Composition Functions (F21-F30).
    """
    cache_key = (func_id, dimension, n_funcs)
    if cache_key in _rotation_matrices_cache:
        return _rotation_matrices_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"M_{func_id}_D{dimension}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"M_{func_id}_D{dimension}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[CEC2017 Error] Rotation matrices missing: '{filepath}'.")

    M = np.loadtxt(filepath)
    matrices = []
    for i in range(n_funcs):
        matrices.append(M[i * dimension: (i + 1) * dimension, :])
        
    _rotation_matrices_cache[cache_key] = matrices
    return matrices


def generate_shift_vectors(func_id: int, dimension: int, n_funcs: int) -> List[np.ndarray]:
    """
    Reads multiple shift vectors from a stacked shift_data_{func_id}.txt file
    """
    cache_key = (func_id, dimension, n_funcs)
    if cache_key in _shift_vectors_cache:
        return _shift_vectors_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"shift_data_{func_id}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"shift_data_{func_id}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[CEC2017 Error] Shift vectors missing: '{filepath}'.")

    shifts = np.loadtxt(filepath)
    vectors = []
    for i in range(n_funcs):
        # We take the first dimension elements of each row
        vectors.append(shifts[i, :dimension])
        
    _shift_vectors_cache[cache_key] = vectors
    return vectors


def generate_shuffle_vectors(func_id: int, dimension: int, n_funcs: int) -> List[np.ndarray]:
    """
    Reads multiple shuffle vectors from a stacked shuffle_data_{func_id}_D{dimension}.txt file
    """
    cache_key = (func_id, dimension, n_funcs)
    if cache_key in _shuffle_vectors_cache:
        return _shuffle_vectors_cache[cache_key]

    filepath = os.path.join(DATA_DIR, f"shuffle_data_{func_id}_D{dimension}.m")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, f"shuffle_data_{func_id}_D{dimension}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[CEC2017 Error] Shuffle vectors missing: '{filepath}'.")

    shuffles = np.loadtxt(filepath, dtype=int) - 1
    vectors = []
    for i in range(n_funcs):
        vectors.append(shuffles[i, :dimension])
        
    _shuffle_vectors_cache[cache_key] = vectors
    return vectors
