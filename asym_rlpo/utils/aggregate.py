from typing import Sequence

import torch


def average(data: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(data, start=torch.tensor(0.0)) / len(data)
