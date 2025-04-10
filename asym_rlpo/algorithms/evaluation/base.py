from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from ..base import Algorithm_ABC


class Evaluation_ABC(Algorithm_ABC):
    @abc.abstractmethod
    def compute_v_values(self, models: nn.ModuleDict, episode: Episode) -> torch.Tensor:
        assert False

    def critic_loss(  # pylint: disable=too-many-locals
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> torch.Tensor:
        if q_estimator is None:
            q_estimator = td0_q_estimator

        v_values = self.compute_v_values(self.models, episode)

        with torch.no_grad():
            target_v_values = self.compute_v_values(self.target_models, episode)
            target_q_values = q_estimator(
                episode.rewards, target_v_values, discount=discount
            )

        critic_loss = F.mse_loss(v_values, target_q_values, reduction='sum')

        return critic_loss
