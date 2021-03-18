import gym
import torch
import torch.nn as nn
from asym_rlpo.utils.debugging import checkraise

from .base import Representation


class IdentityRepresentation(Representation, nn.Module):
    def __init__(self, input_space: gym.spaces.Box):
        super().__init__()

        checkraise(
            isinstance(input_space, gym.spaces.Box)
            and len(input_space.shape) == 1,
            TypeError,
            'input_space must be Box',
        )

        (self.__out_dim,) = input_space.shape

    @property
    def dim(self):
        return self.__out_dim

    def forward(  # pylint: disable=no-self-use
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        return inputs
