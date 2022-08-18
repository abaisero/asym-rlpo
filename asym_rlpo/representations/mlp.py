from typing import Sequence

import torch

from asym_rlpo.modules.mlp import make_mlp

from .base import Representation


class MLPRepresentation(Representation):
    def __init__(self, dims: Sequence[int]):
        super().__init__()

        self.dims = dims
        nonlinearities = ['relu'] * (len(dims) - 1)
        self.model = make_mlp(self.dims, nonlinearities)

    @property
    def dim(self):
        return self.dims[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
