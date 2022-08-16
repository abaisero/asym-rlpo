import itertools as itt
from typing import Sequence

import gym
import torch

from asym_rlpo.modules.mlp import make_mlp
from asym_rlpo.utils.debugging import checkraise

from .base import Representation


class MLPRepresentation(Representation):
    def __init__(self, input_space: gym.spaces.Box, dims: Sequence[int]):
        super().__init__()

        checkraise(
            isinstance(input_space, gym.spaces.Box)
            and len(input_space.shape) == 1,
            TypeError,
            'input_space must be Box',
        )
        checkraise(
            len(dims) > 0,
            ValueError,
            'dims must be non-empty',
        )

        (input_dim,) = input_space.shape
        self.dims = list(itt.chain([input_dim], dims))
        nonlinearities = ['relu'] * len(dims)
        self.model = make_mlp(self.dims, nonlinearities)

    @property
    def dim(self):
        return self.dims[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
