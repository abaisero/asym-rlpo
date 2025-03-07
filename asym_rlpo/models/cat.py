from collections.abc import Sequence

import torch

from asym_rlpo.models.model import Model


class CatModel(Model):
    def __init__(self, representations: Sequence[Model]):
        super().__init__()
        self.representations = representations

    @property
    def dim(self):
        return sum(representation.dim for representation in self.representations)

    def forward(self, inputs):
        return torch.cat(
            [representation(inputs) for representation in self.representations],
            dim=-1,
        )
