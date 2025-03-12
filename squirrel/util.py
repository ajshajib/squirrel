"""This module contains class and functions for general use."""

from scipy import linalg as la
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
    # Step 1: Make the matrix symmetric
    b = (matrix + matrix.T) / 2

    # Step 2: Perform Singular Value Decomposition (SVD)
    _, s, v = la.svd(b)

    # Step 3: Construct the symmetric positive semi-definite matrix
    h = np.dot(v.T, np.dot(np.diag(s), v))

    # Step 4: Average b and h to get a new symmetric matrix
    matrix_2 = (b + h) / 2

    # Step 5: Ensure the matrix is symmetric
    matrix_3 = (matrix_2 + matrix_2.T) / 2

    # Step 6: Check if the matrix is positive-definite
    if is_positive_definite(matrix_3):
        return matrix_3

    # Step 7: If not, adjust the matrix to make it positive-definite
    spacing = np.spacing(la.norm(matrix))
    identity = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(matrix_3):
        min_eigenvalue = np.min(np.real(la.eigvals(matrix_3)))
        matrix_3 += identity * (-min_eigenvalue * k**2 + spacing)
        k += 1

    return matrix_3


def is_positive_definite(matrix):
    """Returns true when input is positive-definite, via Cholesky.

    This function attempts to perform a Cholesky decomposition of the
    input matrix. If the decomposition is successful, the matrix is
    positive-definite. If it fails, the matrix is not positive-definite.

    :param matrix: matrix to check
    :type matrix: numpy.ndarray
    :return: True if matrix is positive-definite
    :rtype: bool
    """
    try:
        _ = la.cholesky(matrix, lower=True)
        return True
    except la.LinAlgError:
        return False
