import numpy as np
from numba import njit


@njit(cache=True)
def dot_3d(matrix_3d, vector):
    """Calculate dot product of 3D matrix and vector. This function is needed because Numba does not yet support dot
    product involving 3D matrix.

    Args:
        matrix_3d (np.ndarray): 3D matrix involved in dot product.
        vector (np.ndarray): Vector involved in dot product.

    Returns:
        np.ndarray: Result of dot product (matrix).
    """
    matrix_3d = matrix_3d.astype(np.float64)

    result = np.empty(matrix_3d.shape[:2], dtype=np.float64)
    for i in range(matrix_3d.shape[0]):
        for j in range(matrix_3d.shape[1]):
            result[i, j] = np.dot(matrix_3d[i, j, :], vector)

    return result


@njit(cache=True)
def get_mirrored_beta(beta):
    """Get mirrored beta if beta is not in interval [0, 1].

    Args:
        beta (float): Beta value that is not in interval [0, 1].

    Returns:
        float: Mirrored beta value in interval [0, 1].
    """
    if 0 <= beta <= 1:
        return beta

    beta = abs(beta)
    beta -= (beta // 2) * 2

    if beta > 1:
        beta = 2 - beta

    if beta < 0:
        beta = abs(beta)

    return beta
