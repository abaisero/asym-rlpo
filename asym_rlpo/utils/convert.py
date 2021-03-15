from typing import Dict, List, overload

import numpy as np
import torch


@overload
def numpy2torch(data: np.ndarray) -> torch.Tensor:
    ...


@overload
def numpy2torch(data: List[np.ndarray]) -> List[torch.Tensor]:
    ...


@overload
def numpy2torch(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    ...


def numpy2torch(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)

    if isinstance(data, list):
        return [numpy2torch(d) for d in data]

    if isinstance(data, dict):
        return {k: numpy2torch(v) for k, v in data.items()}

    raise TypeError(f'unsupported data type {type(data)}')
