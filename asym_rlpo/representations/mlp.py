import gym
import torch
import torch.nn as nn
from asym_rlpo.utils.debugging import checkraise

from .base import Representation


class MLPRepresentation(Representation, nn.Module):
    def __init__(self, input_space: gym.spaces.Box, dim: int):
        super().__init__()

        checkraise(
            isinstance(input_space, gym.spaces.Box)
            and len(input_space.shape) == 1,
            TypeError,
            'input_space must be Box',
        )

        self.input_space = input_space
        (self.__in_dim,) = input_space.shape
        self.__out_dim = dim
        # TODO use some kind of batch norm?
        # self.model = nn.Sequential(
        #     nn.Linear(self.__in_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.__out_dim),
        # )
        self.model = nn.Linear(self.__in_dim, self.__out_dim)

    @property
    def dim(self):
        return self.__out_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
