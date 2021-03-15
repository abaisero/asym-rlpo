import numpy as np
import pytest
from asym_rlpo.utils.collate import collate


@pytest.mark.parametrize(
    'data,expected',
    [
        (
            [0, 1, 2, 3, 4, 5],
            np.array([0, 1, 2, 3, 4, 5]),
        ),
        (
            [
                np.array([0, 1]),
                np.array([2, 3]),
                np.array([4, 5]),
            ],
            np.array([[0, 1], [2, 3], [4, 5]]),
        ),
        (
            [
                {'a': np.array([0, 1]), 'b': np.array([2, 3])},
                {'a': np.array([4, 5]), 'b': np.array([6, 7])},
                {'a': np.array([8, 9]), 'b': np.array([10, 11])},
            ],
            {
                'a': np.array([[0, 1], [4, 5], [8, 9]]),
                'b': np.array([[2, 3], [6, 7], [10, 11]]),
            },
        ),
    ],
)
def test_collate(data, expected):
    np.testing.assert_equal(collate(data), expected)
