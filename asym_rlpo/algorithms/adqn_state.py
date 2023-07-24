from __future__ import annotations

import torch
import torch.nn as nn

from asym_rlpo.algorithms.algorithm import ValueBasedAlgorithm
from asym_rlpo.algorithms.losses import dqn_loss_all_values, dqn_loss_bootstrap
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.qmodel import QhaModel, QzaModel
from asym_rlpo.types import LossDict


class ADQN_State(ValueBasedAlgorithm):
    def __init__(
        self,
        qha_model: QhaModel,
        qza_model: QzaModel,
        target_qha_model: QhaModel,
        target_qza_model: QzaModel,
        *,
        trainer: Trainer,
    ):
        models = nn.ModuleDict(
            {
                'qha_model': qha_model,
                'qza_model': qza_model,
                'target_qha_model': target_qha_model,
                'target_qza_model': target_qza_model,
            }
        )
        super().__init__(models, trainer)

        self.qha_model = qha_model
        self.qza_model = qza_model
        self.target_qha_model = target_qha_model
        self.target_qza_model = target_qza_model

    def target_pairs(
        self,
    ) -> list[tuple[QhaModel, QhaModel] | tuple[QzaModel, QzaModel]]:
        return [
            (self.target_qha_model, self.qha_model),
            (self.target_qza_model, self.qza_model),
        ]

    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        qha_values = self.qha_model.values(episode)
        qza_values = self.qza_model.values(episode)

        with torch.no_grad():
            target_qha_values = self.target_qha_model.values(episode)
            target_qza_values = self.target_qza_model.values(episode)

        return {
            'qha': dqn_loss_bootstrap(
                qha_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qza_values,
                target_qha_values,
            ),
            'qza': dqn_loss_bootstrap(
                qza_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qza_values,
                target_qha_values,
            ),
        }


class ADQN_State_VarianceReduced(ADQN_State):
    def compute_losses(self, episode: Episode, *, discount: float) -> LossDict:
        qha_values = self.qha_model.values(episode)
        qza_values = self.qza_model.values(episode)

        with torch.no_grad():
            target_qha_values = self.target_qha_model.values(episode)
            target_qza_values = self.target_qza_model.values(episode)

        return {
            'qha': dqn_loss_all_values(qha_values, target_qza_values),
            'qza': dqn_loss_bootstrap(
                qza_values,
                episode.actions,
                episode.rewards,
                discount,
                target_qza_values,
                target_qha_values,
            ),
        }
