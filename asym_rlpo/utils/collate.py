from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import torch


@overload
def collate_numpy(data: Sequence[np.ndarray]) -> np.ndarray:
    ...


@overload
def collate_numpy(
    data: Sequence[dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    ...


@overload
def collate_numpy(data: Sequence[Sequence[Any]]) -> np.ndarray:
    ...


def collate_numpy(data):
    if isinstance(data[0], np.ndarray):
        return np.stack(data)

    if isinstance(data[0], dict):
        return {k: collate_numpy([d[k] for d in data]) for k in data[0].keys()}

    try:
        return np.array(data)
    except TypeError as e:
        raise TypeError(f'unsupported data type {type(data[0])}') from e


@overload
def collate_torch(data: Sequence[torch.Tensor]) -> torch.Tensor:
    ...


@overload
def collate_torch(
    data: Sequence[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    ...


@overload
def collate_torch(data: Sequence[Sequence[Any]]) -> torch.Tensor:
    ...


def collate_torch(data):
    if isinstance(data[0], torch.Tensor):
        return torch.stack(data)

    if isinstance(data[0], dict):
        return {k: collate_torch([d[k] for d in data]) for k in data[0].keys()}

    try:
        return torch.tensor(data)
    except TypeError as e:
        raise TypeError(f'unsupported data type {type(data[0])}') from e
