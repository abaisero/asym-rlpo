from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.algorithms.algorithm import Algorithm
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.actor_critic import MemoryReactive_ActorCriticModel
from asym_rlpo.models.critic import HM_CriticModel
from asym_rlpo.q_estimators import Q_Estimator
from asym_rlpo.types import LossDict


class MemoryReactive_A2C(Algorithm):
    def __init__(
        self,
        actor_critic_model: MemoryReactive_ActorCriticModel,
        target_critic_model: HM_CriticModel,
        trainer: Trainer,
    ):
        models = nn.ModuleDict(
            {
                'actor_critic_model': actor_critic_model,
                'target_critic_model': target_critic_model,
            }
        )
        super().__init__(models, trainer)

        self.actor_critic_model = actor_critic_model
        self.target_critic_model = target_critic_model

    def target_pairs(self) -> list[tuple[HM_CriticModel, HM_CriticModel]]:
        return [
            (
                self.target_critic_model,
                self.actor_critic_model.critic_model,
            )
        ]

    def compute_losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Q_Estimator,
    ) -> LossDict:
        action_logits = self.actor_critic_model.actor_model.action_logits(
            episode
        )
        v_values = self.actor_critic_model.critic_model.values(episode)
        device = action_logits.device

        with torch.no_grad():
            q_values = q_estimator(
                episode.rewards,
                v_values.detach(),
                discount=discount,
            )
            advantages = q_values - v_values

            target_v_values = self.target_critic_model.max_memory_values(
                episode
            )
            target_q_values = q_estimator(
                episode.rewards,
                target_v_values,
                discount=discount,
            )

        # policy loss
        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        policy_loss = (discounts * advantages * action_nlls).sum()

        # negentropy loss
        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        # critic loss
        critic_loss = F.mse_loss(v_values, target_q_values, reduction='sum')

        return {
            'policy': policy_loss,
            'negentropy': negentropy_loss,
            'critic': critic_loss,
        }
