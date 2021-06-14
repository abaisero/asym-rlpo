from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_O, Torch_S
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

    @staticmethod
    def compute_features_and_values(
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
        states: Torch_S,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        history_features = compute_history_features(
            models, actions, observations
        )
        state_features = models.state_model(states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        vhs_values = models.vhs_model(inputs).squeeze(-1)
        return history_features, vhs_values

    def losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> LossesDict:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        history_features, vhs_values = self.compute_features_and_values(
            self.models,
            episode.actions,
            episode.observations,
            episode.states,
        )
        qhs_values = q_estimator(episode.rewards, vhs_values, discount=discount)
        action_logits = self.models.policy_model(history_features)

        with torch.no_grad():
            _, target_vhs_values = self.compute_features_and_values(
                self.target_models,
                episode.actions,
                episode.observations,
                episode.states,
            )
            target_qhs_values = q_estimator(
                episode.rewards, target_vhs_values, discount=discount
            )

        discounts = discount ** torch.arange(len(episode), device=self.device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = qhs_values.detach() - vhs_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        critic_loss = F.mse_loss(vhs_values, target_qhs_values, reduction='sum')

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return {
            'actor': actor_loss,
            'critic': critic_loss,
            'negentropy': negentropy_loss,
        }
