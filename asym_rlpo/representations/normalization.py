import torch
import torch.nn as nn

from .base import Representation


class SaneBatchNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # batchnorm does not support arbitrary batch dimensions
        shape = inputs.shape
        inputs = inputs.reshape(-1, inputs.size(-1))

        try:
            outputs = self.batchnorm(inputs)
        except ValueError:
            training = self.batchnorm.training
            self.batchnorm.eval()
            outputs = self.batchnorm(inputs)
            self.batchnorm.train(training)

        return outputs.reshape(shape)


class NormalizationRepresentation(Representation):
    def __init__(self, representation: Representation):
        super().__init__()
        self._representation = representation
        self._sanebatchnorm = SaneBatchNorm1d(representation.dim, affine=False)

    @property
    def dim(self):
        return self._representation.dim

    def forward(self, *args, **kwargs):
        features = self._representation(*args, **kwargs)

        if not isinstance(features, tuple):
            features = self._sanebatchnorm(features)
        else:
            # HistoryRepresentation case
            features, hidden = features
            features = self._sanebatchnorm(features)
            features = features, hidden

        return features

    def __getattr__(self, name):
        # has to be done via super bc of how torch modules are `registered`
        return super().__getattr__(name)
