from numbers import Number
from typing import Type

import numpy as np
import torch


def checkraise(
    condition: bool,
    error_type: Type[Exception],
    error_message_fmt: str,
    *args,
    **kwargs
):
    if not condition:
        raise error_type(error_message_fmt.format(*args, **kwargs))


# taken from https://stackoverflow.com/questions/18376935/best-practice-for-equality-in-python
def nested_equal(a, b):
    """
    Compare two objects recursively by element, handling numpy objects.

    Assumes hashable items are not mutable in a way that affects equality.
    """

    if type(a) is not type(b):
        return False

    if isinstance(a, str):
        return a == b

    if isinstance(a, Number):
        return a == b

    if isinstance(a, np.ndarray):
        return np.all(a == b)

    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)

    if isinstance(a, list):
        return all(nested_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False

        return all(nested_equal(a[k], b[k]) for k in a.keys())

    return a == b
