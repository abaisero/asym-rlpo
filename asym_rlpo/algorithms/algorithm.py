from __future__ import annotations

import abc

import torch.nn as nn

from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.qmodel import QhaModel
from asym_rlpo.types import LossDict
from asym_rlpo.utils.target_update_functions import TargetPair


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, models: nn.ModuleDict, trainer: Trainer):
        super().__init__()
        self.models = models
        self.trainer = trainer

    def state_dict(self):
        return {
            'models': self.models.state_dict(),
            'trainer': self.trainer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.models.load_state_dict(state_dict['models'])
        self.trainer.load_state_dict(state_dict['trainer'])

    @abc.abstractmethod
    def target_pairs(self) -> list[TargetPair]:
        assert False


class ValueBasedAlgorithm(Algorithm):
    qha_model: QhaModel

    @abc.abstractmethod
    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        assert False
