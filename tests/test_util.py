import numpy as np
import pytest
from squirrel.util import get_nearest_positive_definite_matrix
from squirrel.util import is_positive_definite


class TestUtil:
    """
    A test class for utility functions in the `squirrel.util` module.
    This class contains tests for the functions `is_positive_definite` and `get_nearest_positive_definite_matrix`.
    """

    def test_is_positive_definite():
        """
        Test the function `is_positive_definite` to check if a given matrix is positive-definite.

        This function will test the behavior of `is_positive_definite` with both positive-definite
        and non-positive-definite matrices.
        """
        # Create a positive-definite matrix
        matrix = np.array([[2, -1], [-1, 2]])
        # Assert that the matrix is positive-definite
        assert is_positive_definite(matrix) is True

        # Create a non-positive-definite matrix
        matrix = np.array([[1, 2], [2, 1]])
        # Assert that the matrix is not positive-definite
        assert is_positive_definite(matrix) is False

    def test_get_nearest_positive_definite_matrix():
        """
        Test the function `get_nearest_positive_definite_matrix` to find the nearest positive-definite matrix.

        This function will test the behavior of `get_nearest_positive_definite_matrix` with non-positive-definite
        matrices and check if the returned matrix is positive-definite and close to the original matrix.
        """
        # Create a non-positive-definite matrix
        matrix = np.array([[1, 2], [2, 1]])
        # Get the nearest positive-definite matrix
        nearest_pd_matrix = get_nearest_positive_definite_matrix(matrix)

        # Check if the original matrix is not positive-definite
        assert is_positive_definite(matrix) is False
        # Check if the nearest matrix is positive-definite
        assert is_positive_definite(nearest_pd_matrix) is True

        # Check if the nearest positive-definite matrix is close to the original matrix
        assert np.allclose(
            nearest_pd_matrix, np.array([[1.5, 1.5], [1.5, 1.5]]), atol=1e-10
        )

        # Create another non-positive-definite matrix
        matrix = np.array([[0, 0], [0, 1]])
        # Get the nearest positive-definite matrix
        nearest_pd_matrix = get_nearest_positive_definite_matrix(matrix)

        # Check if the original matrix is not positive-definite
        assert is_positive_definite(matrix) is False
        # Check if the nearest matrix is positive-definite
        assert is_positive_definite(nearest_pd_matrix) is True

        # Check if the nearest positive-definite matrix is close to the original matrix
        assert np.allclose(nearest_pd_matrix, np.array([[0, 0], [0, 1]]), atol=1e-10)


if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main()
