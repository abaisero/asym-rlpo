from __future__ import annotations

from typing import cast, Callable

import torch
import torch.nn.functional as F

from asym_rlpo.algorithms.a2c import A2C
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.actor import ActorModel
from asym_rlpo.models.critic import CriticModel, CriticModels, HZ_CriticModel
from asym_rlpo.q_estimators import Q_Estimator
from asym_rlpo.types import LossDict

CounterfactualSampler = Callable[[Episode], torch.Tensor]


def make_counterfactual_episode(
    episode: Episode,
    counterfactual_sampler: CounterfactualSampler,
) -> Episode:
    counterfactual_latents = counterfactual_sampler(episode)
    return Episode(
        observations=episode.observations,
        latents=counterfactual_latents,
        actions=episode.actions,
        rewards=episode.rewards,
        info=episode.info,
    )


class CounterfactualA2C(A2C):
    def __init__(
        self,
        actor_model: ActorModel,
        critic_models: CriticModels,
        target_critic_models: CriticModels,
        trainer: Trainer,
        *,
        counterfactual_sampler: CounterfactualSampler,
    ):
        super().__init__(
            actor_model=actor_model,
            critic_models=critic_models,
            target_critic_models=target_critic_models,
            trainer=trainer,
        )

        self.counterfactual_sampler = counterfactual_sampler

    def compute_losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Q_Estimator,
    ) -> tuple[LossDict, LossDict]:
        actor_model = self.actor_model
        critic_model = cast(CriticModel, self.critic_models['critic'])
        assert isinstance(critic_model, HZ_CriticModel)
        target_critic_model = cast(CriticModel, self.target_critic_models['critic'])
        assert isinstance(target_critic_model, HZ_CriticModel)

        action_logits = actor_model.action_logits(episode)
        v_values = critic_model.values(episode)
        device = action_logits.device

        with torch.no_grad():
            counterfactual_episode = make_counterfactual_episode(
                episode, self.counterfactual_sampler
            )
            counterfactual_v_values = critic_model.values(counterfactual_episode)
            counterfactual_q_values = q_estimator(
                counterfactual_episode.rewards,
                counterfactual_v_values,
                discount=discount,
            )
            counterfactual_advantages = (
                counterfactual_q_values - counterfactual_v_values
            )

            target_v_values = target_critic_model.values(episode)
            target_q_values = q_estimator(
                episode.rewards,
                target_v_values,
                discount=discount,
            )

        # policy loss
        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(1, episode.actions.unsqueeze(-1))
        action_nlls = action_nlls.squeeze(-1)
        policy_loss = (discounts * counterfactual_advantages * action_nlls).sum()

        # negentropy loss
        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        # critic losses
        critic_loss = F.mse_loss(v_values, target_q_values, reduction='sum')

        actor_losses = {
            'policy': policy_loss,
            'negentropy': negentropy_loss,
        }
        critic_losses = {'critic': critic_loss}
        return actor_losses, critic_losses
