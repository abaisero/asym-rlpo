import abc

import torch
import torch.nn as nn


class Model(nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def dim(self):
        assert False


class FeatureModel(Model):
    @abc.abstractmethod
    def zeros_like(self, device: torch.device | None = None):
        assert False
