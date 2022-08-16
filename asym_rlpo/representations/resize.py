from asym_rlpo.modules.mlp import make_mlp

from .base import Representation


class ResizeRepresentation(Representation):
    def __init__(self, representation: Representation, dim: int):
        super().__init__()
        self._representation = representation
        self._resize_model = make_mlp([representation.dim, dim], ['relu'])
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
