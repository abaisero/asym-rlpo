import torch.nn as nn

from .base import Representation


class NormalizationRepresentation(Representation, nn.Module):
    def __init__(self, representation: Representation):
        super().__init__()
        self.representation = representation
        self.normalization_model = nn.LayerNorm(representation.dim)

    @property
    def dim(self):
        return self.representation.dim

    def forward(self, *args, **kwargs):
        features = self.representation(*args, **kwargs)

        # handles HistoryRepresentation
        if isinstance(features, tuple):
            features, hidden = features
            features = self.normalization_model(features)
            features = features, hidden
        else:
            features = self.normalization_model(features)

        return features
