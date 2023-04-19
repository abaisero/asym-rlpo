import abc

import torch

from asym_rlpo.data import TorchObservation
from asym_rlpo.types import Features


class HistoryIntegrator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self, observation: TorchObservation):
        assert False

    @abc.abstractmethod
    def step(self, action: torch.Tensor, observation: TorchObservation):
        assert False

    @abc.abstractmethod
    def sample_features(self) -> tuple[Features, dict]:
        assert False
