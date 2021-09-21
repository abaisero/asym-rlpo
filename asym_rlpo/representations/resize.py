import torch.nn as nn

from asym_rlpo.modules import make_module

from .base import Representation


class ResizeRepresentation(Representation):
    def __init__(self, representation: Representation, dim: int):
        super().__init__()
        self._representation = representation
        self._resize_model = nn.Sequential(
            make_module('linear', 'relu', representation.dim, dim),
            nn.ReLU(),
        )
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def forward(self, *args, **kwargs):
        features = self._representation(*args, **kwargs)

        # handles HistoryRepresentation
        if isinstance(features, tuple):
            features, hidden = features
            features = self._resize_model(features)
            features = features, hidden
        else:
            features = self._resize_model(features)

        return features

    def __getattr__(self, name):
        # has to be done via super bc of how torch modules are `registered`
        return super().__getattr__(name)
