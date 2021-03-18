from numbers import Number
from typing import Dict, Sequence, TypeVar, Union, overload

import numpy as np

T = TypeVar('T', np.ndarray, Dict[str, np.ndarray])


@overload
def collate(data: Sequence[Union[bool, int, float]]) -> np.ndarray:
    ...


@overload
def collate(data: Sequence[Sequence[Union[bool, int, float]]]) -> np.ndarray:
    ...


@overload
def collate(data: Sequence[T]) -> T:
    ...


def collate(data):
    if np.isscalar(data[0]):
        return np.array(data)

    if isinstance(data[0], tuple):
        return np.array(data)

    if isinstance(data[0], np.ndarray):
        return np.stack(data)

    if isinstance(data[0], dict):
        return {k: collate([d[k] for d in data]) for k in data[0].keys()}

    raise TypeError(f'unsupported data type {type(data[0])}')
