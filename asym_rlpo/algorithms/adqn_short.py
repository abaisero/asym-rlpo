from __future__ import annotations

import torch
import torch.nn as nn

from asym_rlpo.algorithms.algorithm import ValueBasedAlgorithm
from asym_rlpo.algorithms.losses import (
    dqn_loss_action_values,
    dqn_loss_all_values,
    dqn_loss_bootstrap,
)
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.qmodel import QhaModel, QhzaModel
from asym_rlpo.types import LossDict


class ADQN_Short(ValueBasedAlgorithm):
    def __init__(
        self,
        qha_model: QhaModel,
        qhza_model: QhzaModel,
        target_qha_model: QhaModel,
        target_qhza_model: QhzaModel,
        *,
        trainer: Trainer,
    ):
        models = nn.ModuleDict(
            {
                'qha_model': qha_model,
                'qhza_model': qhza_model,
                'target_qha_model': target_qha_model,
                'target_qhza_model': target_qhza_model,
            }
        )
        super().__init__(models, trainer)

        self.qha_model = qha_model
        self.qhza_model = qhza_model
        self.target_qha_model = target_qha_model
        self.target_qhza_model = target_qhza_model

    def target_pairs(
        self,
    ) -> list[tuple[QhaModel, QhaModel] | tuple[QhzaModel, QhzaModel]]:
        return [
            (self.target_qha_model, self.qha_model),
            (self.target_qhza_model, self.qhza_model),
        ]

    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        qha_values = self.qha_model.values(episode)
        qhza_values = self.qhza_model.values(episode)

        with torch.no_grad():
            target_qha_values = self.target_qha_model.values(episode)
            target_qhza_values = self.target_qhza_model.values(episode)

        return {
            'qha': dqn_loss_bootstrap(
                qha_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qhza_values,
                target_qha_values,
            ),
            'qhza': dqn_loss_action_values(
                qhza_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qha_values,
            ),
        }


class ADQN_Short_VarianceReduced(ADQN_Short):
    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        qha_values = self.qha_model.values(episode)
        qhza_values = self.qhza_model.values(episode)

        with torch.no_grad():
            target_qha_values = self.target_qha_model.values(episode)
            target_qhza_values = self.target_qhza_model.values(episode)

        return {
            'qha': dqn_loss_all_values(qha_values, target_qhza_values),
            'qhza': dqn_loss_action_values(
                qhza_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qha_values,
            ),
        }
