import torch.nn as nn

from .base import Representation


class ResizeRepresentation(Representation, nn.Module):
    def __init__(self, representation: Representation, dim: int):
        super().__init__()
        self.representation = representation
        self.resize_model = nn.Linear(representation.dim, dim, bias=False)
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def forward(self, *args, **kwargs):
        features = self.representation(*args, **kwargs)

        # handles HistoryRepresentation
        if isinstance(features, tuple):
            features, hidden = features
            features = self.resize_model(features)
            features = features, hidden
        else:
            features = self.resize_model(features)

        return features
