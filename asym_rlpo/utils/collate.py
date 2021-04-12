from typing import Dict, Sequence, TypeVar, Union, overload

import numpy as np
import torch

T = TypeVar('T', np.ndarray, Dict[str, np.ndarray])


@overload
def collate_numpy(data: Sequence[Union[bool, int, float]]) -> np.ndarray:
    ...


@overload
def collate_numpy(
    data: Sequence[Sequence[Union[bool, int, float]]]
) -> np.ndarray:
    ...


@overload
def collate_numpy(data: Sequence[T]) -> T:
    ...


def collate_numpy(data):
    if np.isscalar(data[0]):
        return np.array(data)

    if isinstance(data[0], tuple):
        return np.array(data)

    if isinstance(data[0], np.ndarray):
        return np.stack(data)

    if isinstance(data[0], dict):
        return {k: collate_numpy([d[k] for d in data]) for k in data[0].keys()}

    raise TypeError(f'unsupported data type {type(data[0])}')


def collate_torch(data):
    if np.isscalar(data[0]):
        return torch.tensor(data)

    if isinstance(data[0], tuple):
        return torch.tensor(data)

    if isinstance(data[0], torch.Tensor):
        return torch.stack(data)

    if isinstance(data[0], dict):
        return {k: collate_torch([d[k] for d in data]) for k in data[0].keys()}

    raise TypeError(f'unsupported data type {type(data[0])}')
