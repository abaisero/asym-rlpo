from numbers import Number
from typing import overload

import numpy as np
import torch


def is_dtype_integer(x: np.ndarray) -> bool:
    """checks if array has an integer type

    Args:
        x (np.ndarray): x

    Returns:
        bool:
    """
    return np.issubdtype(x.dtype, np.integer)


def is_dtype_floating(x: np.ndarray) -> bool:
    """checks if array has a floating type

    Args:
        x (np.ndarray): x

    Returns:
        bool:
    """
    return np.issubdtype(x.dtype, np.floating)


def is_dtype_boolean(x: np.ndarray) -> bool:
    """checks if array has a boolean type

    Args:
        x (np.ndarray): x

    Returns:
        bool:
    """
    return np.issubdtype(x.dtype, bool)


@overload
def numpy2torch(data: None) -> None:
    ...


@overload
def numpy2torch(data: int | float | bool) -> torch.Tensor:
    ...


@overload
def numpy2torch(data: np.ndarray) -> torch.Tensor:
    ...


@overload
def numpy2torch(data: list[np.ndarray]) -> list[torch.Tensor]:
    ...


@overload
def numpy2torch(data: tuple[np.ndarray]) -> tuple[torch.Tensor]:
    ...


@overload
def numpy2torch(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    ...


def numpy2torch(data):
    if data is None:
        return None

    if isinstance(data, Number):
        return torch.tensor(data)

    if isinstance(data, np.ndarray):
        return (
            torch.from_numpy(data).float()
            if is_dtype_floating(data)
            else torch.from_numpy(data)
        )

    if isinstance(data, list):
        return [numpy2torch(d) for d in data]

    if isinstance(data, tuple):
        return tuple(numpy2torch(d) for d in data)

    if isinstance(data, dict):
        return {k: numpy2torch(v) for k, v in data.items()}

    raise TypeError(f'unsupported data type {type(data)}')
