from __future__ import annotations

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.algorithms.algorithm import Algorithm
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.envs.utils.beliefs import compute_beliefs
from asym_rlpo.models.actor_critic import NoisyActorCriticModel
from asym_rlpo.models.critic import CriticModel, H_CriticModel, HZ_CriticModel
from asym_rlpo.q_estimators import Q_Estimator
from asym_rlpo.types import LossDict


class NoisyA2C(Algorithm):
    def __init__(
        self,
        actor_critic_model: NoisyActorCriticModel,
        target_h_critic_model: H_CriticModel,
        target_hz_critic_model: HZ_CriticModel,
        trainer: Trainer,
    ):
        models = nn.ModuleDict(
            {
                'actor_critic_model': actor_critic_model,
                'target_h_critic_model': target_h_critic_model,
                'target_hz_critic_model': target_hz_critic_model,
            }
        )
        super().__init__(models, trainer)

        self.actor_critic_model = actor_critic_model
        self.target_h_critic_model = target_h_critic_model
        self.target_hz_critic_model = target_hz_critic_model

    def target_pairs(
        self,
    ) -> list[tuple[CriticModel, CriticModel]]:
        return [
            (
                self.target_h_critic_model,
                self.actor_critic_model.h_critic_model,
            ),
            (
                self.target_hz_critic_model,
                self.actor_critic_model.hz_critic_model,
            ),
        ]

    def compute_losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Q_Estimator,
        pomdp: gym.Env,
    ) -> LossDict:
        action_logits = self.actor_critic_model.actor_model.action_logits(episode)
        vh_values = self.actor_critic_model.h_critic_model.values(episode)
        vhz_values = self.actor_critic_model.hz_critic_model.values(episode)
        device = action_logits.device

        with torch.no_grad():
            target_vh_values = self.target_h_critic_model.values(episode)
            target_qh_values = q_estimator(
                episode.rewards,
                target_vh_values,
                discount=discount,
            )

            target_vhz_values = self.target_hz_critic_model.values(episode)
            target_qhz_values = q_estimator(
                episode.rewards,
                target_vhz_values,
                discount=discount,
            )

            noise_variance = compute_hz_values_variance(
                episode,
                hz_critic_model=self.target_hz_critic_model,
                pomdp=pomdp,
            )
            # (T,)
            noise = torch.randn_like(target_vh_values) * noise_variance.sqrt()
            noisy_v_values = target_vh_values + noise
            noisy_q_values = q_estimator(
                episode.rewards, noisy_v_values.detach(), discount=discount
            )
            noisy_advantages = noisy_q_values - noisy_v_values

        # policy loss
        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        policy_loss = (discounts * noisy_advantages * action_nlls).sum()

        # negentropy loss
        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        # critic losses
        h_critic_loss = F.mse_loss(vh_values, target_qh_values, reduction='sum')
        hz_critic_loss = F.mse_loss(vhz_values, target_qhz_values, reduction='sum')

        return {
            'policy': policy_loss,
            'negentropy': negentropy_loss,
            'h_critic': h_critic_loss,
            'hz_critic': hz_critic_loss,
        }


def compute_hz_values_variance(
    episode: Episode,
    *,
    hz_critic_model: HZ_CriticModel,
    pomdp: gym.Env,
) -> torch.Tensor:
    beliefs = compute_beliefs(episode, pomdp=pomdp)

    history_features = hz_critic_model.history_model.episodic(episode)
    # (T, Dh)
    latent_features = hz_critic_model.latent_model.embeddings.weight.detach()
    # (S, Dz)

    T, Dh = history_features.shape
    S, Dz = latent_features.shape
    assert S == beliefs.shape[-1]

    history_features = history_features.unsqueeze(1).expand(-1, S, -1)
    latent_features = latent_features.unsqueeze(0).expand(T, -1, -1)
    input_features = torch.cat([history_features, latent_features], dim=-1)
    # (T, S, Dh+Dl)
    assert input_features.shape == (T, S, Dh + Dz)
    hz_values = hz_critic_model.value_module(input_features).squeeze(-1)
    # (T, S)

    first_moment = torch.einsum('ts,ts->t', hz_values, beliefs)
    # (T,)
    squared_deviation = (hz_values - first_moment.unsqueeze(-1)) ** 2
    # (T, S)
    variance = torch.einsum('ts,ts->t', squared_deviation, beliefs)
    # (T,)

    # # numerically unstable, can result in negative variances
    # second_moment = (hz_values**2 * beliefs).sum(-1)
    # # (T,)
    # first_moment_squared = (hz_values * beliefs).sum(-1) ** 2
    # # (T,)
    # variance = second_moment - first_moment_squared
    # # numerical errors may cause small negative variances
    #
    # assert variance.min() >= -1e-6
    # return variance + torch.minimum(variance, torch.zeros_like(variance))

    return variance
