import torch
import torch.nn as nn

from .base import Representation


class SaneBatchNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            return self.batchnorm(inputs)
        except ValueError:
            training = self.batchnorm.training
            self.batchnorm.eval()
            outputs = self.batchnorm(inputs)
            self.batchnorm.train(training)
            return outputs


class NormalizationRepresentation(Representation, nn.Module):
    def __init__(self, representation: Representation):
        super().__init__()
        self.representation = representation
        self.sanebatchnorm = SaneBatchNorm1d(representation.dim, affine=False)

    @property
    def dim(self):
        return self.representation.dim

    def forward(self, *args, **kwargs):
        features = self.representation(*args, **kwargs)

        try:
            # handles HistoryRepresentation
            features, hidden = features
        except ValueError:
            # normal usecase
            features = self.sanebatchnorm(features)
        else:
            features = self.sanebatchnorm(features)
            features = features, hidden

        return features
