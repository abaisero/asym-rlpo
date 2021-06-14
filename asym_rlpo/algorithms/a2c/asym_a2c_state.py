from typing import Optional

import torch
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from .base import A2C_Base, LossesDict


class AsymA2C_State(A2C_Base):
    model_keys = [
        'state_model',
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vs_model',
    ]

    def losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None
    ) -> LossesDict:
        if q_estimator is None:
            q_estimator = td0_q_estimator

        history_features = compute_history_features(
            self.models, episode.actions, episode.observations
        )

        action_logits = self.models.policy_model(history_features)
        state_features = self.models.state_model(episode.states)
        vs_values = self.models.vs_model(state_features).squeeze(-1)
        qs_targets = q_estimator(
            episode.rewards, vs_values.detach(), discount=discount
        )

        discounts = discount ** torch.arange(len(episode), device=self.device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = qs_targets - vs_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vs_values, qs_targets, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
