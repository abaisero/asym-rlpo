from typing import Optional

import torch
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from .base import A2C_Base, LossesDict


class A2C(A2C_Base):
    model_keys = [
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vh_model',
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
        vh_values = self.models.vh_model(history_features).squeeze(-1)
        qh_targets = q_estimator(
            episode.rewards, vh_values.detach(), discount=discount
        )

        discounts = discount ** torch.arange(len(episode), device=self.device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = qh_targets - vh_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vh_values, qh_targets, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
