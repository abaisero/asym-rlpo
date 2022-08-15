import gym
import torch

from asym_rlpo.utils.debugging import checkraise

from .base import Representation


class IdentityRepresentation(Representation):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
