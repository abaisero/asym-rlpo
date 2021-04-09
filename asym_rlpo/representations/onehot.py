import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.utils.debugging import checkraise

from .base import Representation


class OneHotRepresentation(Representation, nn.Module):
    def __init__(self, input_space: gym.spaces.Discrete):
        super().__init__()

        checkraise(
            isinstance(input_space, gym.spaces.Discrete),
            TypeError,
            'input_space must be Discrete',
        )

        self.__num_classes = input_space.n

    @property
    def dim(self):
        return self.__num_classes

    def forward(  # pylint: disable=no-self-use
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        return F.one_hot(inputs, num_classes=self.__num_classes).float()
