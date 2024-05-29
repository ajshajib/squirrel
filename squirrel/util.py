"""This module contains class and functions for general use."""

from numpy import linalg as la
import numpy as np


def get_nearest_positive_definite_matrix(matrix):
    """Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Adapted from https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd

    :param matrix: matrix to find nearest positive-definite matrix to
    :type matrix: numpy.ndarray
    :return: nearest positive-definite matrix
    :rtype: numpy.ndarray
    """

    b = (matrix + matrix.T) / 2
    _, s, v = la.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))

    matrix_2 = (b + h) / 2

    matrix_3 = (matrix_2 + matrix_2.T) / 2

    if is_positive_definite(matrix_3):
        return matrix_3

    spacing = np.spacing(la.norm(matrix))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    identity = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(matrix_3):
        min_eigenvalue = np.min(np.real(la.eigvals(matrix_3)))
        matrix_3 += identity * (-min_eigenvalue * k**2 + spacing)
        k += 1

    return matrix_3


def is_positive_definite(matrix):
    """Returns true when input is positive-definite, via Cholesky.

    :param matrix: matrix to check
    :type matrix: numpy.ndarray
    :return: True if matrix is positive-definite
    :rtype: bool
    """
    try:
        _ = la.cholesky(matrix)
        return True
    except la.LinAlgError:
        return False
