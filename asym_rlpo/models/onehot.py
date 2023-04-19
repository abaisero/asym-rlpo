import gym.spaces
import torch
import torch.nn.functional as F

from asym_rlpo.models.model import FeatureModel


class OneHotModel(FeatureModel):
    def __init__(self, space: gym.spaces.Discrete):
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError(
                f'Invalid space type; should be gym.spaces.Discrete, is {type(space)}'
            )

        super().__init__()

        self.__num_classes = space.n

    @property
    def dim(self):
        return self.__num_classes

    def forward(self, inputs):
        return F.one_hot(inputs, num_classes=self.__num_classes).float()

    def zeros_like(self, device: torch.device | None = None):
        return torch.zeros(self.__num_classes, device=device)
