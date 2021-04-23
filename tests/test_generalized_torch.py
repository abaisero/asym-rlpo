import pytest
import torch

import asym_rlpo.generalized_torch as gtorch


def assert_equal(x, y):
    assert isinstance(x, dict) == isinstance(y, dict)

    if isinstance(x, dict):
        assert x.keys() == y.keys()
        assert all(torch.equal(x[k], y[k]) for k in x.keys())
        assert all(x[k].dtype == y[k].dtype for k in x.keys())
    else:
        assert torch.equal(x, y)
        assert x.dtype == y.dtype


@pytest.mark.parametrize(
    'data,expected',
    [
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 0, 0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0.0, 0.0, 0.0]),
        ),
        (
            {'x': torch.tensor([1, 2, 3]), 'y': torch.tensor([4, 5])},
            {'x': torch.tensor([0, 0, 0]), 'y': torch.tensor([0, 0])},
        ),
        (
            {'x': torch.tensor([1.0, 2.0, 3.0]), 'y': torch.tensor([4.0, 5.0])},
            {'x': torch.tensor([0.0, 0.0, 0.0]), 'y': torch.tensor([0.0, 0.0])},
        ),
    ],
)
def test_zeros_like(data, expected):
    assert_equal(gtorch.zeros_like(data), expected)


@pytest.mark.parametrize(
    'data,dim,expected',
    [
        (
            torch.tensor([[1, 2, 3]]),
            0,
            torch.tensor([1, 2, 3]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0]]),
            0,
            torch.tensor([1.0, 2.0, 3.0]),
        ),
        (
            {
                'x': torch.tensor([[1, 2, 3]]),
                'y': torch.tensor([[4, 5]]),
            },
            0,
            {
                'x': torch.tensor([1, 2, 3]),
                'y': torch.tensor([4, 5]),
            },
        ),
        (
            {
                'x': torch.tensor([[1.0, 2.0, 3.0]]),
                'y': torch.tensor([[4.0, 5.0]]),
            },
            0,
            {
                'x': torch.tensor([1.0, 2.0, 3.0]),
                'y': torch.tensor([4.0, 5.0]),
            },
        ),
    ],
)
def test_squeeze(data, dim, expected):
    assert_equal(gtorch.squeeze(data, dim), expected)


@pytest.mark.parametrize(
    'data,dim,expected',
    [
        (
            torch.tensor([1, 2, 3]),
            0,
            torch.tensor([[1, 2, 3]]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            0,
            torch.tensor([[1.0, 2.0, 3.0]]),
        ),
        (
            {
                'x': torch.tensor([1, 2, 3]),
                'y': torch.tensor([4, 5]),
            },
            0,
            {
                'x': torch.tensor([[1, 2, 3]]),
                'y': torch.tensor([[4, 5]]),
            },
        ),
        (
            {
                'x': torch.tensor([1.0, 2.0, 3.0]),
                'y': torch.tensor([4.0, 5.0]),
            },
            0,
            {
                'x': torch.tensor([[1.0, 2.0, 3.0]]),
                'y': torch.tensor([[4.0, 5.0]]),
            },
        ),
    ],
)
def test_unsqueeze(data, dim, expected):
    assert_equal(gtorch.unsqueeze(data, dim), expected)
