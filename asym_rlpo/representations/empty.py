import torch

from .base import Representation


class EmptyRepresentation(Representation):
    @property
    def dim(self):
        return 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape + (0,))
