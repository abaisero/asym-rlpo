from __future__ import annotations

import torch
import torch.nn as nn

from asym_rlpo.algorithms.algorithm import ValueBasedAlgorithm
from asym_rlpo.algorithms.losses import dqn_loss_action_values
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.qmodel import QhaModel
from asym_rlpo.types import LossDict


class DQN(ValueBasedAlgorithm):
    def __init__(
        self,
        qha_model: QhaModel,
        target_qha_model: QhaModel,
        *,
        trainer: Trainer,
    ):
        models = nn.ModuleDict(
            {
                'qha_model': qha_model,
                'target_qha_model': target_qha_model,
            }
        )
        super().__init__(models, trainer)

        self.qha_model = qha_model
        self.target_qha_model = target_qha_model

    def update_target_parameters(self):
        self.target_qha_model.load_state_dict(self.qha_model.state_dict())

    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        qha_values = self.qha_model.values(episode)

        with torch.no_grad():
            target_qha_values = self.target_qha_model.values(episode)

        return {
            'qha': dqn_loss_action_values(
                qha_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qha_values,
            )
        }
