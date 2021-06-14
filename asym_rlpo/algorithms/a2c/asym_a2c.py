from typing import Optional

import torch
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from .base import A2C_Base, LossesDict


class AsymA2C(A2C_Base):
    model_keys = [
        'state_model',
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vhs_model',
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
        inputs = torch.cat([history_features, state_features], dim=-1)
        vhs_values = self.models.vhs_model(inputs).squeeze(-1)
        qhs_targets = q_estimator(
            episode.rewards, vhs_values.detach(), discount=discount
        )

        discounts = discount ** torch.arange(len(episode), device=self.device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = qhs_targets - vhs_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vhs_values, qhs_targets, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
