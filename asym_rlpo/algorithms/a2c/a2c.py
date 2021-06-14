from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_O, Torch_S
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

    @staticmethod
    def compute_features_and_values(
        models: nn.ModuleDict, actions: torch.Tensor, observations: Torch_O
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        history_features = compute_history_features(
            models, actions, observations
        )
        vh_values = models.vh_model(history_features).squeeze(-1)
        return history_features, vh_values

    def losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> LossesDict:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        history_features, vh_values = self.compute_features_and_values(
            self.models,
            episode.actions,
            episode.observations,
        )
        qh_values = q_estimator(episode.rewards, vh_values, discount=discount)
        action_logits = self.models.policy_model(history_features)

        with torch.no_grad():
            _, target_vh_values = self.compute_features_and_values(
                self.target_models,
                episode.actions,
                episode.observations,
            )
            target_qh_values = q_estimator(
                episode.rewards, target_vh_values, discount=discount
            )

        discounts = discount ** torch.arange(len(episode), device=self.device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = qh_values.detach() - vh_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vh_values, target_qh_values, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
