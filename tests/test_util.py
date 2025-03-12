import numpy as np
import pytest
from squirrel.util import get_nearest_positive_definite_matrix
from squirrel.util import is_positive_definite


def test_is_positive_definite():
    # Create a positive-definite matrix
    matrix = np.array([[2, -1], [-1, 2]])
    assert is_positive_definite(matrix) == True

    # Create a non-positive-definite matrix
    matrix = np.array([[1, 2], [2, 1]])
    assert is_positive_definite(matrix) == False


def test_get_nearest_positive_definite_matrix():
    # Create a non-positive-definite matrix
    matrix = np.array([[1, 2], [2, 1]])
    nearest_pd_matrix = get_nearest_positive_definite_matrix(matrix)

    # Check if the result is positive-definite
    assert is_positive_definite(nearest_pd_matrix) == True

    # Check if the result is close to the original matrix
    assert np.allclose(nearest_pd_matrix, np.array([[1.5, 1.5], [1.5, 1.5]]), atol=1e-1)


if __name__ == "__main__":
    pytest.main()
