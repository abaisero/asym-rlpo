import itertools as itt
from typing import Sequence

import gym
import more_itertools as mitt
import torch
import torch.nn as nn

from asym_rlpo.modules import make_module
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

        modules = mitt.flatten(
            (make_module('linear', 'relu', in_dim, out_dim), nn.ReLU())
            for in_dim, out_dim in mitt.pairwise(self.dims)
        )
        self.model = nn.Sequential(*modules)

    @property
    def dim(self):
        return self.dims[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
