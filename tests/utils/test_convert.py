import numpy as np
import pytest
import torch
import torch.testing

from asym_rlpo.utils.convert import numpy2torch


@pytest.mark.parametrize(
    'data,expected',
    [
        (
            0,
            torch.tensor(0),
        ),
        (
            np.array([0, 1, 2, 3, 4, 5]),
            torch.tensor([0, 1, 2, 3, 4, 5]),
        ),
        (
            [
                np.array([0, 1]),
                np.array([2, 3]),
                np.array([4, 5]),
            ],
            [
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
                torch.tensor([4, 5]),
            ],
        ),
        (
            [
                {'a': np.array([0, 1]), 'b': np.array([2, 3])},
                {'a': np.array([4, 5]), 'b': np.array([6, 7])},
                {'a': np.array([8, 9]), 'b': np.array([10, 11])},
            ],
            [
                {'a': torch.tensor([0, 1]), 'b': torch.tensor([2, 3])},
                {'a': torch.tensor([4, 5]), 'b': torch.tensor([6, 7])},
                {'a': torch.tensor([8, 9]), 'b': torch.tensor([10, 11])},
            ],
        ),
    ],
)
def test_numpy2torch(data, expected):
    torch.testing.assert_close(numpy2torch(data), expected, rtol=0.0, atol=0.0)
