from __future__ import annotations

from typing import Callable, cast

import torch
import torch.nn.functional as F

from asym_rlpo.algorithms.a2c import A2C
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.actor import ActorModel
from asym_rlpo.models.critic import CriticModel, CriticModels
from asym_rlpo.q_estimators import Q_Estimator
from asym_rlpo.types import LossDict

PhaseIdentifier = Callable[[Episode], torch.Tensor]


class BiphasicA2C(A2C):
    def __init__(
        self,
        actor_model: ActorModel,
        critic_models: CriticModels,
        target_critic_models: CriticModels,
        trainer: Trainer,
        *,
        phase_identifier: PhaseIdentifier,
    ):
        super().__init__(
            actor_model=actor_model,
            critic_models=critic_models,
            target_critic_models=target_critic_models,
            trainer=trainer,
        )
        self.phase_identifier = phase_identifier

    def compute_losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Q_Estimator,
    ) -> tuple[LossDict, LossDict]:
        actor_model = self.actor_model
        critic_model_0 = cast(CriticModel, self.critic_models['critic_0'])
        critic_model_1 = cast(CriticModel, self.critic_models['critic_1'])
        target_critic_model_0 = cast(CriticModel, self.target_critic_models['critic_0'])
        target_critic_model_1 = cast(CriticModel, self.target_critic_models['critic_1'])

        action_logits = actor_model.action_logits(episode)
        v_values_0 = critic_model_0.values(episode)
        v_values_1 = critic_model_1.values(episode)
        device = action_logits.device

        with torch.no_grad():
            q_values_0 = q_estimator(
                episode.rewards,
                v_values_0.detach(),
                discount=discount,
            )
            advantages_0 = q_values_0 - v_values_0

            target_v_values_0 = target_critic_model_0.values(episode)
            target_q_values_0 = q_estimator(
                episode.rewards,
                target_v_values_0,
                discount=discount,
            )

            q_values_1 = q_estimator(
                episode.rewards,
                v_values_1.detach(),
                discount=discount,
            )
            advantages_1 = q_values_1 - v_values_1

            target_v_values_1 = target_critic_model_1.values(episode)
            target_q_values_1 = q_estimator(
                episode.rewards,
                target_v_values_1,
                discount=discount,
            )

            phases = self.phase_identifier(episode)
            advantages = torch.where(phases == 0, advantages_0, advantages_1)

        # policy loss
        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(1, episode.actions.unsqueeze(-1))
        action_nlls = action_nlls.squeeze(-1)
        policy_loss = (discounts * advantages * action_nlls).sum()

        # negentropy loss
        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        # critic losses
        critic_loss_0 = F.mse_loss(v_values_0, target_q_values_0, reduction='sum')
        critic_loss_1 = F.mse_loss(v_values_1, target_q_values_1, reduction='sum')

        actor_losses = {
            'policy': policy_loss,
            'negentropy': negentropy_loss,
        }
        critic_losses = {'critic_0': critic_loss_0, 'critic_1': critic_loss_1}
        return actor_losses, critic_losses
