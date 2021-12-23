from typing import Sequence

import torch

from .base import Representation


class CatRepresentation(Representation):
    def __init__(self, representations: Sequence[Representation]):
        super().__init__()
        self.representations = representations

    @property
    def dim(self):
        return sum(
            representation.dim for representation in self.representations
        )

    def forward(self, inputs):
        return torch.cat(
            [representation(inputs) for representation in self.representations],
            dim=-1,
        )
