import gym
import gym.spaces
import torch

from asym_rlpo.models.model import Model


class IdentityModel(Model):
    def __init__(self, space: gym.spaces.Box):
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f'Invalid space type; should be gym.spaces.Box, is {type(space)}'
            )

        if space.shape is None or len(space.shape) != 1:
            raise ValueError(
                f'Invalid space shape;  should have single dimension, has {space.shape}'
            )

        super().__init__()
        (self.__out_dim,) = space.shape

    @property
    def dim(self):
        return self.__out_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
