from typing import Dict

import torch
import torch.nn.functional as F

from asym_rlpo.data import Episode

from .base import A2C, LossesDict


class SymA2C(A2C):
    model_keys = frozenset(
        [
            'action_model',
            'observation_model',
            'history_model',
            'policy_model',
            'vh_model',
        ]
    )

    def losses(self, episode: Episode, *, discount: float) -> LossesDict:
        action_features = self.models.action_model(episode.actions)
        action_features = action_features.roll(1, 0)
        action_features[0, :] = 0.0
        observation_features = self.models.observation_model(
            episode.observations
        )

        inputs = torch.cat([action_features, observation_features], dim=-1)
        history_features, _ = self.models.history_model(inputs.unsqueeze(0))
        history_features = history_features.squeeze(0)

        action_logits = self.models.policy_model(history_features)
        vh_values = self.models.vh_model(history_features).squeeze(-1)
        vh_values_bootstrap = torch.tensor(0.0).where(
            episode.dones, vh_values.detach().roll(-1)
        )
        vh_targets = episode.rewards + discount * vh_values_bootstrap

        discounts = discount ** torch.arange(len(episode))
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = vh_targets - vh_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vh_values, vh_targets, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
