from collections.abc import Sequence

import torch

from asym_rlpo.types import LossDict


def average(data: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(data, start=torch.tensor(0.0)) / len(data)


def average_losses(losses: Sequence[LossDict]) -> LossDict:
    return {k: average([loss[k] for loss in losses]) for k in losses[0].keys()}
